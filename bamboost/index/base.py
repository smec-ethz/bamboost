"""
Indexing of bamboost collections and their simulations/parameters. SQLAlchemy is used to
interact with the SQLite database.

The index is generated on the fly or can be explicitly created by scanning the
`search_paths` for collections. The index is stored as a SQLite database that stores the
path of collections (characterized with a unique UID), as well as the metadata and
parameters of all simulations.

The `bamboost.index.base.Index` class provides the public API for interacting with the
index. This works in paralell execution, but the class is designed to execute any
operations on the database on the root process only. Methods that return something use
`bcast` to cast the result to all processes. Any SQL operation is executed only on the
root process!

Database schema:
- `collections`: Contains information about the collections, namely uids and corresponding
  paths.
- `simulations`: Contains information about the simulations, including names, statuses,
  and links to the corresponding parameters.
- `parameters`: Contains the parameters associated with the simulations.
"""

from __future__ import annotations

import fnmatch
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

import yaml
from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from typing_extensions import Concatenate

from bamboost import BAMBOOST_LOGGER, config, constants
from bamboost._typing import _P, _T, SimulationMetadataT, SimulationParameterT, StrPath
from bamboost.exceptions import InvalidCollectionError
from bamboost.index import store
from bamboost.index.schema import collections_table, simulations_table
from bamboost.mpi import Communicator
from bamboost.mpi.utilities import RootProcessMeta
from bamboost.utilities import PathSet

if TYPE_CHECKING:
    from bamboost.mpi import Comm

log = BAMBOOST_LOGGER.getChild("Database")

IDENTIFIER_PREFIX = ".bamboost-collection"
IDENTIFIER_SEPARATOR = "-"


class CollectionUID(str):
    """UID of a collection. If no UID is provided, a new one is generated."""

    def __new__(cls, uid: Optional[str] = None, length: int = 10):
        uid = uid or cls.generate_uid(length)
        return super().__new__(cls, uid.upper())

    @staticmethod
    def generate_uid(length: int) -> str:
        if Communicator._active_comm.rank == 0:
            uid = uuid.uuid4().hex[:length].upper()
        else:
            uid = ""
        uid: str = Communicator._active_comm.bcast(uid, root=0)
        return uid


def _sql_transaction(
    func: Callable[Concatenate[Index, _P], _T],
) -> Callable[Concatenate[Index, _P], _T]:
    """Decorator to add a session to the function signature.

    Args:
        func: The function to decorate.
    """

    @wraps(func)
    def inner(self: Index, *args: _P.args, **kwargs: _P.kwargs) -> Any:
        with self.sql_transaction():
            return func(self, *args, **kwargs)

    return inner


class LazyDefaultIndex:
    def __init__(self) -> None:
        self._instance = None

    def __get__(self, instance: None, owner: Type[Index]) -> Index:
        assert instance is None, (
            "The default index is a class attribute! Use `Index.default` instead."
        )
        if self._instance is None:
            self._instance = Index(
                sql_file=config.index.databaseFile,
                search_paths=config.index.searchPaths,
            )
        return self._instance

    def __set__(self, instance: None, value: Index) -> None:
        self._instance = value

    def __delete__(self, instance: None) -> None:
        raise AttributeError("Cannot delete LazyDefaultIndex descriptor.")


class Index(metaclass=RootProcessMeta):
    """API for indexing BAMBOOST collections and simulations.

    Usage:
        Create an instance of the `Index` class and use its methods to interact
        with the index.
        >>> from bamboost.index import Index
        >>> index = Index()

        Scan for collections in known paths:
        >>> index.scan_for_collections()

        Resolve the path of a collection:
        >>> index.resolve_path(<collection-uid>)

        Get a simulation from its collection and simulation name:
        >>> index.get_simulation(<collection-uid>, <simulation-name>)

    Args:
        cache: CacheAPI instance to use for the index. If not provided, a new
            instance is created with the default cache file.
    """

    _comm = Communicator()
    _engine: Engine
    _sm: Callable[..., Session]
    _s: Session
    search_paths: PathSet

    default: ClassVar[LazyDefaultIndex] = LazyDefaultIndex()
    """A default index instance. Uses the default SQLite database file and search paths
    from the configuration."""

    def __init__(
        self,
        sql_file: Optional[StrPath] = None,
        comm: Optional[Comm] = None,
        *,
        search_paths: Optional[Iterable[str | Path]] = None,
    ) -> None:
        self.search_paths = PathSet(search_paths or config.index.searchPaths)
        """Paths to scan for collections."""

        self._file = sql_file or config.index.databaseFile
        """The path to the SQLite database file."""

        self._isolated = config.index.isolated
        """Whether project based indexing is used."""

        self._url = f"sqlite:///{self._file}"
        """The URL to the SQLite database file."""

        self._initialize_root_process(self._url)

    def _initialize_root_process(self, url: str) -> None:
        self._engine = create_engine(
            url,
            json_serializer=store.json_serializer,
            json_deserializer=store.json_deserializer,
        )

        def _fk_pragma_on_connect(dbapi_con, _con_record):
            dbapi_con.execute("pragma foreign_keys=ON")

        event.listen(self._engine, "connect", _fk_pragma_on_connect)
        store.create_all(self._engine)
        self._sm = sessionmaker(
            bind=self._engine, autobegin=False, expire_on_commit=False
        )
        self._s = self._sm()

    @RootProcessMeta.exclude
    @contextmanager
    def sql_transaction(self) -> Generator[Session, None, None]:
        """Context manager for a SQL transaction.

        If no transaction is active, a new transaction is started. If a
        transaction is active, the current session is used.

        Usage:
            >>> with index.sql_transaction() as s:
            ...     s.execute(...)
        """
        # if not root rank, return dummy context manager
        if not self._comm.rank == 0:
            yield None  # type: ignore
            return

        if self._s.in_transaction():
            try:
                yield self._s
            except SQLAlchemyError as e:
                log.warning(f"Caching transaction failed: {e}")
            return

        try:
            self._s.begin()
            yield self._s
            self._s.commit()
        except SQLAlchemyError as e:
            log.warning(f"Caching transaction failed: {e}")
            self._s.rollback()
            raise
        finally:
            self._s.close()  # Not decided yet if we should close the session

    @_sql_transaction
    def scan_for_collections(
        self,
        *,
        search_paths: Optional[PathSet] = None,
    ) -> list[tuple[str, Path]]:
        """Scan known paths for collections and update the index.

        Iterates through the search paths and searches files with the
        identifier file structure. If a collection is found, it is added to the
        cache.

        Args:
            search_paths (List[Path], optional): Paths to scan for collections.
                Defaults to config.index.searchPaths.
        """
        search_paths = PathSet(search_paths) or self.search_paths
        all_found_collections = []

        for path in search_paths:
            found_collections: tuple[tuple[str, Path], ...] = (
                _scan_directory_for_collections(path)
            )
            if not found_collections:
                continue
            collections_data: list[dict[str, Any]] = []
            for found_uid, found_path in found_collections:
                normalized_uid = found_uid.upper()
                record: dict[str, Any] = {
                    "uid": normalized_uid,
                    "path": found_path.as_posix(),
                }
                metadata = load_collection_metadata(found_path, normalized_uid)
                if metadata is not None:
                    record.update(metadata)
                collections_data.append(record)
            self._s.execute(store.collections_upsert_stmt(collections_data))
            log.debug(f"Inserting found collections:\n{collections_data}")
            all_found_collections.extend(found_collections)

        return all_found_collections

    @_sql_transaction
    def check_integrity(self) -> None:
        """Check the integrity of the cache.

        This method checks if the paths stored in the cache are valid. If a
        path is not valid, it is removed from the cache.
        """
        rows = self._s.execute(
            select(collections_table.c.uid, collections_table.c.path)
        ).all()
        for uid, path in rows:
            if not _validate_path(Path(path), uid):
                log.info(
                    "Invalid collection path in cache: %s -> removing.",
                    (uid, path),
                )
                self._s.execute(store.delete_collection_stmt(uid))

    def _get_collection_record(self, identifier: str) -> store.CollectionRecord | None:
        if not identifier:
            return None

        normalized_uid = identifier.upper()
        collection = store.fetch_collection(self._s, normalized_uid)
        if collection is not None:
            return collection

        alias_uid = self._find_collection_uid_by_alias(identifier)
        if alias_uid is None:
            return None

        return store.fetch_collection(self._s, alias_uid)

    def _find_collection_uid_by_alias(self, alias: str) -> str | None:
        return store.fetch_collection_uid_by_alias(self._s, alias)

    @RootProcessMeta.bcast_result
    @_sql_transaction
    def resolve_path(
        self,
        uid: str,
        *,
        search_paths: Optional[Set[StrPath]] = None,
    ) -> Path:
        """Resolve and return the path of a collection from its UID. Raises a
        `FileNotFoundError` if the collection is not found in the search paths.

        Args:
            uid: UID of the collection
            search_paths: Paths to search for the collection

        Raises:
            FileNotFoundError: If the collection is not found in the search paths
        """
        collection = self._get_collection_record(uid)
        stored_path = (
            Path(collection.path) if collection else self._get_collection_path(uid)
        )
        target_uid = collection.uid if collection else uid.upper()

        if stored_path and _validate_path(stored_path, target_uid):
            return stored_path

        log.debug(
            "No or invalid path found in cache for collection <%s>.",
            uid,
        )

        # Try to find the collection in the search paths
        for root_dir in PathSet(search_paths) or self.search_paths:
            log.debug(f"Searching for collection <{uid}> in <{root_dir}>")
            paths_found = _find_collection(target_uid, Path(root_dir))

            if len(paths_found) > 0:  # If at least one file is found
                if len(paths_found) > 1:
                    log.warning(
                        f"Multiple collections found for {uid}. Using the first one."
                        f"\n{paths_found}"
                    )

                # Store the collection in the cache
                found_path = paths_found[0]
                self.upsert_collection(target_uid, found_path)
                return found_path

        raise FileNotFoundError(f"Database with {uid} was not found.")

    @RootProcessMeta.bcast_result
    @_sql_transaction
    def resolve_uid(self, path: StrPath) -> CollectionUID:
        """Resolve the UID of a collection from a path.

        Returns the UID of the collection or a new UID if it can't be
        determined.

        Args:
            path: Path of the collection

        Raises:
            NotACollectionError: If not collection is found at the given path.
        """
        path = Path(path)
        cached_uid: str | None = self._s.execute(
            select(collections_table.c.uid).where(
                collections_table.c.path == path.as_posix()
            )
        ).scalar()
        if cached_uid and _validate_path(path, cached_uid):
            return CollectionUID(cached_uid)

        log.debug(f"No or invalid uid found in cache for collection <{path}>.")

        identified_uid = _find_uid_from_path(path)
        if identified_uid:
            return CollectionUID(identified_uid)
        else:
            raise InvalidCollectionError("No collection found at the given path.")

    @_sql_transaction
    def sync_collection(
        self, uid: str, path: Optional[StrPath] = None, *, force_all: bool = False
    ) -> None:
        """Sync the table with the file system.

        Iterates through the simulations in the collection and updates the
        metadata and parameters if the HDF5 file has been modified.

        Args:
            uid: UID of the collection
            path (Optional): Path of the collection
        """
        path = Path(path or self.resolve_path(uid)).absolute()
        # Get all simulation names in the file system
        all_simulations_fs = set(
            (
                i.name
                for i in path.iterdir()
                if i.is_dir() and i.joinpath(constants.HDF_DATA_FILE_NAME).is_file()
            )
        )

        collection = store.fetch_collection(self._s, uid)

        if collection:
            for simulation in collection.simulations:
                if simulation.name not in all_simulations_fs:
                    self._s.execute(store.delete_simulation_stmt(uid, simulation.name))
                    continue

                # if the HDF5 file has not been modified since the last sync,
                # remove the simulation from the active update set
                if force_all:
                    continue
                h5_file = path.joinpath(simulation.name, constants.HDF_DATA_FILE_NAME)
                if (  # type: ignore
                    datetime.fromtimestamp(h5_file.stat().st_mtime)
                    <= simulation.modified_at
                ):
                    all_simulations_fs.remove(simulation.name)

        for name in all_simulations_fs:
            log.debug(f"Syncing simulation {name} in collection {uid}.")
            self.upsert_simulation(
                collection_uid=uid, simulation_name=name, collection_path=path
            )

    @property
    @RootProcessMeta.bcast_result
    @_sql_transaction
    def all_collections(self) -> Sequence[store.CollectionRecord]:
        """Return all collections in the index."""
        return store.fetch_collections(self._s)

    @RootProcessMeta.bcast_result
    @_sql_transaction
    def collection(self, uid: str) -> store.CollectionRecord | None:
        """Return a collection from the index.

        Args:
            uid: UID of the collection
        """
        log.debug("Fetching collection from cache.")
        return self._get_collection_record(uid)

    @property
    @RootProcessMeta.bcast_result
    @_sql_transaction
    def all_simulations(self) -> Sequence[store.SimulationRecord]:
        """Return all simulations in the index."""
        return store.fetch_simulations(self._s)

    @RootProcessMeta.bcast_result
    @_sql_transaction
    def simulation(
        self, collection_uid: str, name: str
    ) -> store.SimulationRecord | None:
        """Return a simulation from the index.

        Args:
            collection_uid: UID of the collection
            name: Name of the simulation
        """
        return store.fetch_simulation(self._s, collection_uid, name)

    @property
    @RootProcessMeta.bcast_result
    @_sql_transaction
    def all_parameters(self) -> Sequence[store.ParameterRecord]:
        """Return all parameters in the index."""
        return store.fetch_parameters(self._s)

    @_sql_transaction
    def _drop_collection(self, uid: str) -> None:
        """Drop a collection from the cache.

        Args:
            uid: UID of the collection
        """
        self._s.execute(store.delete_collection_stmt(uid))

    @_sql_transaction
    def _drop_simulation(self, collection_uid: str, simulation_name: str) -> None:
        """Drop a simulation from the cache.

        Args:
            collection_uid: UID of the collection
            simulation_name: Name of the simulation
        """
        self._s.execute(store.delete_simulation_stmt(collection_uid, simulation_name))

    @_sql_transaction
    def upsert_collection(
        self, uid: str, path: Path, metadata: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Cache a collection in the index.

        Args:
            uid: UID of the collection
            path: Path of the collection
        """
        record: dict[str, Any] = {"uid": uid.upper(), "path": path.as_posix()}
        normalized_uid = record["uid"]
        metadata_mapping = metadata or load_collection_metadata(path, normalized_uid)
        if metadata_mapping is not None:
            record.update(_normalize_collection_metadata(metadata_mapping))

        self._s.execute(store.collections_upsert_stmt(record))

    @_sql_transaction
    def upsert_simulation(
        self,
        collection_uid: str,
        simulation_name: str,
        parameters: Optional[Mapping[Any, Any]] = None,
        metadata: Optional[Mapping[Any, Any]] = None,
        *,
        collection_path: Optional[StrPath] = None,
    ) -> None:
        """Cache a simulation from a collection.

        Args:
            collection_uid: UID of the collection
            simulation_name: Name of the simulation
            collection_path (Optional): Path of the collection
        """
        collection_path = Path(collection_path or self.resolve_path(collection_uid))

        if metadata is None and parameters is None:
            from bamboost.core.simulation.base import Simulation

            # if neither metadata nor parameters are provided, read them from the HDF5 file
            # temp change the communicator to MPI.COMM_SELF -> because here is only
            # executed on rank 0
            # 04.03.25: unclear whether passing comm=MPI.COMM_SELF is sufficient in itself
            # 26.03.25: removed the context manager, because it is included in the metaclass
            sim = Simulation(
                simulation_name,
                collection_path,
                index=self,
                collection_uid=collection_uid,
            )
            with sim._file.open("r"):
                metadata, parameters = sim.metadata._dict, sim.parameters._dict

        # Upsert the simulation table
        sim_payload = {
            "collection_uid": collection_uid,
            "name": simulation_name,
            "modified_at": datetime.now(),
            **(metadata or {}),
        }
        sim_id = self._s.execute(
            store.simulations_upsert_stmt(sim_payload)
        ).scalar_one()

        # Upsert the parameters table
        if parameters:
            parameter_payload = [
                {"simulation_id": sim_id, "key": key, "value": value}
                for key, value in parameters.items()
            ]
            self._s.execute(store.parameters_upsert_stmt(parameter_payload))

    @_sql_transaction
    def update_simulation_metadata(
        self, collection_uid: str, simulation_name: str, data: Mapping
    ) -> None:
        """Update the metadata of a simulation by passing it as a dict.

        Args:
            data: Dictionary with new data
        """
        payload = {"collection_uid": collection_uid, "name": simulation_name, **data}
        self._s.execute(store.simulations_upsert_stmt(payload))

    @_sql_transaction
    def update_simulation_parameters(
        self,
        collection_uid: str,
        simulation_name: str,
        parameters: SimulationParameterT,
    ) -> None:
        """Update the parameters of a simulation by passing it as a dict.

        Args:
            parameters: Dictionary with new parameters
        """
        sim_id = self._s.execute(
            select(simulations_table.c.id).where(
                simulations_table.c.collection_uid == collection_uid,
                simulations_table.c.name == simulation_name,
            )
        ).scalar_one()

        parameter_payload = [
            {"simulation_id": sim_id, "key": key, "value": value}
            for key, value in parameters.items()
        ]
        self._s.execute(store.parameters_upsert_stmt(parameter_payload))

    @RootProcessMeta.bcast_result
    @_sql_transaction
    def _get_collection_path(
        self,
        uid: str,
    ) -> Optional[Path]:
        res = self._s.execute(
            select(collections_table.c.path).where(
                collections_table.c.uid == uid.upper()
            )
        ).scalar()
        if res:
            return Path(res)

        alias_uid = self._find_collection_uid_by_alias(uid)
        if alias_uid:
            alias_path = self._s.execute(
                select(collections_table.c.path).where(
                    collections_table.c.uid == alias_uid
                )
            ).scalar()
            if alias_path:
                return Path(alias_path)
        return None

    @RootProcessMeta.bcast_result
    @_sql_transaction
    def _get_collections(self) -> Sequence[store.CollectionRecord]:
        return store.fetch_collections(self._s)

    @RootProcessMeta.bcast_result
    @_sql_transaction
    def _get_simulation(
        self, collection_uid: CollectionUID | str, simulation_name: str
    ) -> store.SimulationRecord | None:
        return store.fetch_simulation(self._s, collection_uid, simulation_name)


def load_collection_metadata(path: Path, uid: str) -> dict[str, Any] | None:
    """Load the metadata of a collection from its identifier file.

    Args:
        path: Path to the collection directory
        uid: UID of the collection
    """
    metadata_file = Path(path).joinpath(get_identifier_filename(uid))
    if not metadata_file.exists():
        return None

    try:
        raw = yaml.safe_load(metadata_file.read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        log.debug(
            "Failed to load metadata for collection %s from %s: %s",
            uid,
            metadata_file,
            exc,
        )
        raise Exception(
            f"Failed to load metadata for collection {uid} (file: {metadata_file})"
        ) from exc

    if raw is None:
        raw = {}

    if not isinstance(raw, Mapping):
        log.debug("Unexpected metadata format for collection %s: %r", uid, raw)
        raise TypeError(
            f"Unexpected metadata format for collection {uid}: {type(raw)}. "
            "Revise the identifier/metadata file."
        )

    return _normalize_collection_metadata(raw)


def _normalize_collection_metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the metadata of a collection.

    This also handles backward compatibility for the "Date of creation" field.

    Args:
        data: The raw metadata dictionary.
    """
    metadata: dict[str, Any] = dict(**data)

    created_at_value = None
    # "Date of creation" is for backward compatibility only
    # we will write "created_at" from now on
    for key in ("created_at", "Date of creation"):
        if key in data and data[key] is not None:
            created_at_value = data[key]
            break

    parsed_created_at = _parse_datetime_value(created_at_value)
    if parsed_created_at is not None:
        metadata["created_at"] = parsed_created_at

    if tags := data.get("tags"):
        metadata["tags"] = _deduplicate_sequence(tags)
    if aliases := data.get("aliases"):
        metadata["aliases"] = _deduplicate_sequence(aliases, casefold=True)

    return metadata


def _deduplicate_sequence(values: Any, *, casefold: bool = False) -> list[str]:
    """Deduplicate a sequence of strings.

    Args:
        values: The sequence of strings to deduplicate.
        casefold: Whether to ignore case when deduplicating.
    """
    if values is None:
        iterable: Iterable[Any] = []
    elif isinstance(values, (str, bytes)):
        iterable = [values]
    else:
        try:
            iterable = list(values)
        except TypeError:
            iterable = [values]

    seen: set[str] = set()
    result: list[str] = []

    for value in iterable:
        text = str(value).strip()
        if not text:
            continue

        key = text.casefold() if casefold else text
        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def _parse_datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None
    return None


def create_identifier_file(path: StrPath, uid: str) -> None:
    """Create an identifier file in the collection directory.

    Args:
        path: Path to the collection directory
        uid: UID of the collection
    """
    path = Path(path)
    with open(path.joinpath(get_identifier_filename(uid)), "w") as f:
        f.write("Date of creation: " + str(datetime.now()))


def get_identifier_filename(uid: str) -> str:
    return IDENTIFIER_PREFIX + IDENTIFIER_SEPARATOR + uid


def _validate_path(path: Path, uid: str) -> bool:
    return path.is_dir() and path.joinpath(get_identifier_filename(uid)).is_file()


def _find_uid_from_path(path: Path) -> Optional[str]:
    try:
        return path.glob(f"{IDENTIFIER_PREFIX}*").__next__().name.rsplit("-", 1)[1]
    except StopIteration:
        return None


def _find_collection(uid: str, root_dir: Path) -> tuple[Path, ...]:
    """Find the collection with UID under given root_dir.

    Args:
        uid: UID to search for
        root_dir: root directory for search
    """
    return tuple(
        Path(i).parent for i in _find_files(get_identifier_filename(uid), root_dir)
    )


def _find_files(
    pattern: str,
    root_dir: str | os.PathLike,
    exclude: Iterable[str] | None = None,
) -> Tuple[Path, ...]:
    """
    Locate every file matching *pattern* under *root_dir* while **pruning**
    directory names listed in *exclude* (exact-match on the final path part).

    Returns an immutable tuple of absolute paths (str) just like the POSIX helper.
    """
    root = Path(root_dir)
    hits: list[Path] = []

    if exclude is None:
        exclude = config.index.excludeDirs

    for base, dirnames, filenames in os.walk(root, topdown=True):
        # --- prune in–place so the walker never descends further ---
        dirnames[:] = [d for d in dirnames if d not in exclude]

        for fname in filenames:
            if fnmatch.fnmatch(fname, pattern):
                hits.append(Path(base, fname))

    return tuple(hits)


def _scan_directory_for_collections(root_dir: Path) -> tuple[tuple[str, Path], ...]:
    """Scan the directory for collections.

    Args:
        root_dir: Directory to scan for collections

    Returns:
        Tuple of tuples with the UID and path of the collection
    """

    log.debug(f"Scanning {root_dir}")

    if not root_dir.exists():
        log.warning(f"Path does not exist: {root_dir}")
        return ()

    found_indicator_files = _find_files(
        get_identifier_filename("*"), root_dir.as_posix()
    )

    if not found_indicator_files:
        log.info(f"No collections found in {root_dir}")
        return ()

    return tuple(
        (i.name.rsplit(IDENTIFIER_SEPARATOR, 1)[-1], i.parent)
        for i in found_indicator_files
    )
