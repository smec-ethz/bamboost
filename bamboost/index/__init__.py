"""Module for indexing BAMBOOST collections.

Uses a caching mechanism using SQLAlchemy and an SQLite database to store
information about collections and simulations.

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

Classes:
    Index: API for indexing BAMBOOST collections and simulations.
"""

from __future__ import annotations

import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from sqlalchemy import Engine, create_engine, delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload, sessionmaker
from typing_extensions import Concatenate

from bamboost import BAMBOOST_LOGGER, config
from bamboost._typing import (
    _P,
    _T,
    StrPath,
    _SimulationMetadataT,
    _SimulationParameterT,
)
from bamboost.index.sqlmodel import (
    CollectionORM,
    ParameterORM,
    SimulationORM,
    create_all,
    json_deserializer,
    json_serializer,
)
from bamboost.mpi import MPI
from bamboost.mpi.utilities import MPISafeMeta, bcast, on_root
from bamboost.utilities import PathSet

if TYPE_CHECKING:
    from bamboost.mpi import Comm

log = BAMBOOST_LOGGER.getChild("Database")

IDENTIFIER_PREFIX = ".BAMBOOST"
IDENTIFIER_SEPARATOR = "-"


class CollectionUID(str):
    """UID of a collection."""

    def __new__(cls, uid: Optional[str] = None, length: int = 10):
        uid = uid or cls.generate_uid(length)
        return super().__new__(cls, uid.upper())

    @staticmethod
    def generate_uid(length: int) -> str:
        return uuid.uuid4().hex[:length].upper()


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


class Index(metaclass=MPISafeMeta):
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

    _comm: Comm
    _engine: Engine
    _sm: Callable[..., Session]
    _s: Session
    search_paths: PathSet

    def __init__(
        self,
        sql_file: Optional[StrPath] = None,
        comm: Optional[Comm] = None,
        *,
        search_paths: Optional[Set[StrPath]] = None,
    ) -> None:
        self._comm = comm or MPI.COMM_WORLD
        self._engine = create_engine(
            f"sqlite:///{sql_file or config.index.databaseFile}",
            json_serializer=json_serializer,
            json_deserializer=json_deserializer,
        )
        on_root(create_all, self._comm)(self._engine)
        self._sm = sessionmaker(
            bind=self._engine, autobegin=False, expire_on_commit=False
        )
        self._s = self._sm()
        self.search_paths = PathSet(search_paths or config.index.searchPaths)

    @contextmanager
    def sql_transaction(self) -> Generator[Session, None, None]:
        """Context manager for a SQL transaction.

        If no transaction is active, a new transaction is started. If a
        transaction is active, the current session is used.

        Usage:
            >>> with index.sql_transaction() as s:
            ...     s.execute(...)
        """
        if self._s.in_transaction():
            yield self._s
            return
        try:
            self._s.begin()
            yield self._s
            self._s.commit()
        except SQLAlchemyError as e:
            # self._s.rollback()  # Is this necessary?
            log.warning(f"Caching transaction failed: {e}")
        finally:
            self._s.close()  # Not decided yet if we should close the session

    @_sql_transaction
    def scan_for_collections(
        self,
        *,
        search_paths: Optional[PathSet] = None,
    ) -> None:
        """Scan known paths for collections and update the index.

        Iterates through the search paths and searches files with the
        identifier file structure. If a collection is found, it is added to the
        cache.

        Args:
            search_paths (List[Path], optional): Paths to scan for collections.
                Defaults to config.index.searchPaths.
        """
        search_paths = PathSet(search_paths) or self.search_paths
        for path in search_paths:
            found_collections: tuple[tuple[str, Path], ...] = (
                _scan_directory_for_collections(path)
            )
            if not found_collections:
                continue
            collections_data = [
                {"uid": uid, "path": str(path)} for uid, path in found_collections
            ]
            self._s.execute(CollectionORM.upsert(collections_data))
            log.info(f"Inserting found collections:\n{found_collections}")

    @_sql_transaction
    def check_integrity(self) -> None:
        """Check the integrity of the cache.

        This method checks if the paths stored in the cache are valid. If a
        path is not valid, it is removed from the cache.
        """
        for collection in self._s.execute(select(CollectionORM)).scalars().all():
            if not _validate_path(Path(collection.path), collection.uid):
                log.warning(
                    f"Invalid collection path in cache: {collection.uid, collection.path} -> removing."
                )
                self._s.delete(collection)

    @bcast
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
        stored_path = self._get_collection_path(uid)

        if stored_path and _validate_path(stored_path, uid):
            return stored_path

        log.debug(f"No or invalid path found in cache for collection <{uid}>.")

        # Try to find the collection in the search paths
        for root_dir in PathSet(search_paths) or self.search_paths:
            log.debug(f"Searching for collection <{uid}> in <{root_dir}>")
            paths_found = _find_collection(uid, Path(root_dir))

            if len(paths_found) > 0:  # If at least one file is found
                if len(paths_found) > 1:
                    log.warning(
                        f"Multiple collections found for {uid}. Using the first one."
                        f"\n{paths_found}"
                    )

                # Store the collection in the cache
                self.upsert_collection(uid, paths_found[0])
                return paths_found[0]

        raise FileNotFoundError(f"Database with {uid} was not found.")

    @bcast
    @_sql_transaction
    def resolve_uid(self, path: StrPath) -> CollectionUID:
        """Resolve the UID of a collection from a path.

        Returns the UID of the collection or a new UID if it can't be
        determined.

        Args:
            path: Path of the collection
        """
        path = Path(path)
        cached_uid: str | None = self._s.execute(
            select(CollectionORM.uid).where(CollectionORM.path == path.as_posix())
        ).scalar()
        if cached_uid and _validate_path(path, cached_uid):
            return CollectionUID(cached_uid)

        log.debug(f"No or invalid path found in cache for collection <{path}>.")

        identified_uid = _find_uid_from_path(path)
        uid = CollectionUID(
            identified_uid
        )  # Note: this generates a new UID if none is found
        self._s.execute(
            CollectionORM.upsert({"uid": uid, "path": path.absolute().as_posix()})
        )
        return uid

    @_sql_transaction
    def sync_collection(self, uid: str, path: Optional[StrPath] = None) -> None:
        """Sync the table with the file system.

        Iterates through the simulations in the collection and updates the
        metadata and parameters if the HDF5 file has been modified.

        Args:
            uid: UID of the collection
            path (Optional): Path of the collection
        """
        path = Path(path or self.resolve_path(uid))
        # Get all simulation names in the file system
        all_simulations_fs = set((i.name for i in path.iterdir() if i.is_dir()))

        collection = self._s.get(CollectionORM, uid)

        if collection:
            for simulation in collection.simulations:
                if simulation.name not in all_simulations_fs:
                    self._s.delete(simulation)
                    continue

                # if the HDF5 file has not been modified since the last sync,
                # remove the simulation from the active update set
                h5_file = path.joinpath(simulation.name, f"{simulation.name}.h5")
                if (  # type: ignore
                    datetime.fromtimestamp(h5_file.stat().st_mtime)
                    <= simulation.modified_at
                ):
                    all_simulations_fs.remove(simulation.name)

        for name in all_simulations_fs:
            self.upsert_simulation(
                collection_uid=uid, simulation_name=name, collection_path=path
            )

    @property
    @bcast
    @_sql_transaction
    def all_collections(self) -> Sequence[CollectionORM]:
        """Return all collections in the index. Eagerly loads the simulations
        and its parameters.
        """
        return (
            self._s.execute(
                select(CollectionORM).options(
                    joinedload(CollectionORM.simulations).subqueryload(
                        SimulationORM.parameters
                    )
                )
            )
            .unique()
            .scalars()
            .all()
        )

    @bcast
    @_sql_transaction
    def collection(self, uid: str) -> CollectionORM | None:
        """Return a collection from the index.

        Args:
            uid: UID of the collection
        """
        log.debug("Fetching collection from cache.")
        return (
            self._s.execute(
                select(CollectionORM)
                .where(CollectionORM.uid == uid)
                .options(
                    joinedload(CollectionORM.simulations).subqueryload(
                        SimulationORM.parameters
                    )
                )
            )
            .unique()
            .scalar()
        )

    @property
    @bcast
    @_sql_transaction
    def all_simulations(self) -> Sequence[SimulationORM]:
        """Return all simulations in the index. Eagerly loads the parameters."""
        return (
            self._s.execute(
                select(SimulationORM).options(joinedload(SimulationORM.parameters))
            )
            .unique()
            .scalars()
            .all()
        )

    @bcast
    @_sql_transaction
    def simulation(self, collection_uid: str, name: str) -> SimulationORM | None:
        """Return a simulation from the index.

        Args:
            collection_uid: UID of the collection
            name: Name of the simulation
        """
        return (
            self._s.execute(
                select(SimulationORM)
                .where(
                    SimulationORM.collection_uid == collection_uid,
                    SimulationORM.name == name,
                )
                .options(joinedload(SimulationORM.parameters))
            )
            .unique()
            .scalar()
        )

    @property
    @bcast
    @_sql_transaction
    def all_parameters(self) -> Sequence[ParameterORM]:
        """Return all parameters in the index."""
        return self._s.execute(select(ParameterORM)).scalars().all()

    @_sql_transaction
    def _drop_collection(self, uid: str) -> None:
        """Drop a collection from the cache.

        Args:
            uid: UID of the collection
        """
        self._s.execute(delete(CollectionORM).where(CollectionORM.uid == uid))

    @_sql_transaction
    def _drop_simulation(self, collection_uid: str, simulation_name: str) -> None:
        """Drop a simulation from the cache.

        Args:
            collection_uid: UID of the collection
            simulation_name: Name of the simulation
        """
        stmt = delete(SimulationORM).where(
            SimulationORM.collection_uid == collection_uid,
            SimulationORM.name == simulation_name,
        )
        self._s.execute(stmt)

    @_sql_transaction
    def upsert_collection(self, uid: str, path: Path) -> None:
        """Cache a collection in the index.

        Args:
            uid: UID of the collection
            path: Path of the collection
        """
        self._s.execute(CollectionORM.upsert({"uid": uid, "path": path.as_posix()}))

    @_sql_transaction
    def upsert_simulation(
        self,
        collection_uid: str,
        simulation_name: str,
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

        h5_file = collection_path.joinpath(simulation_name, f"{simulation_name}.h5")
        metadata, parameters = simulation_metadata_from_h5(h5_file)

        # Upsert the simulation table
        sim_id = self._s.execute(
            SimulationORM.upsert(
                {"collection_uid": collection_uid, "name": simulation_name, **metadata}
            )
        ).scalar_one()

        # Upsert the parameters table
        self._s.execute(
            ParameterORM.upsert(
                [
                    {"simulation_id": sim_id, "key": k, "value": v}
                    for k, v in parameters.items()
                ]
            )
        )

    @_sql_transaction
    def update_simulation_metadata(
        self, collection_uid: str, simulation_name: str, data: _SimulationMetadataT
    ) -> None:
        """Update the metadata of a simulation by passing it as a dict.

        Args:
            data: Dictionary with new data
        """
        self._s.execute(
            SimulationORM.upsert(
                {"collection_uid": collection_uid, "name": simulation_name, **data}
            )
        )

    @_sql_transaction
    def update_simulation_parameters(
        self,
        collection_uid: str,
        simulation_name: str,
        parameters: _SimulationParameterT,
    ) -> None:
        """Update the parameters of a simulation by passing it as a dict.

        Args:
            parameters: Dictionary with new parameters
        """
        sim_id = self._s.execute(
            select(SimulationORM.id).where(
                SimulationORM.collection_uid == collection_uid,
                SimulationORM.name == simulation_name,
            )
        ).scalar_one()

        self._s.execute(
            ParameterORM.upsert(
                [
                    {"simulation_id": sim_id, "key": k, "value": v}
                    for k, v in parameters.items()
                ]
            )
        )

    @bcast
    @_sql_transaction
    def _get_collection_path(
        self,
        uid: str,
    ) -> Optional[Path]:
        res = self._s.execute(
            select(CollectionORM.path).where(CollectionORM.uid == uid)
        ).scalar()
        return Path(res) if res else None

    @bcast
    @_sql_transaction
    def _get_collections(self) -> Sequence[CollectionORM]:
        return self._s.execute(select(CollectionORM)).scalars().all()

    @bcast
    @_sql_transaction
    def _get_simulation(
        self, collection_uid: CollectionUID | str, simulation_name: str
    ) -> SimulationORM | None:
        return self._s.execute(
            select(SimulationORM)
            .where(
                SimulationORM.collection_uid == collection_uid,
                SimulationORM.name == simulation_name,
            )
            .options(joinedload(SimulationORM.parameters))
        ).scalar()


def simulation_metadata_from_h5(
    file: Path,
) -> Tuple[_SimulationMetadataT, _SimulationParameterT]:
    """Extract metadata and parameters from a BAMBOOST simulation HDF5 file.

    Reads the metadata and parameters from the HDF5 file and returns them as a
    tuple.

    Args:
        file: Path to the HDF5 file.
    """
    if not file.is_file():
        raise FileNotFoundError(f"File not found: {file}")

    from bamboost.core.hdf5.file import HDF5File

    with HDF5File(file.as_posix()).open("r") as f:
        meta: _SimulationMetadataT = {
            "created_at": datetime.fromisoformat(f.attrs.get("time_stamp", 0)),
            "modified_at": datetime.fromtimestamp(file.stat().st_mtime),
            "description": f.attrs.get("notes", ""),
            "status": f.attrs.get("status", ""),
        }
        params: _SimulationParameterT = dict(f["parameters"].attrs)

        return meta, params


def create_identifier_file(path: StrPath, uid: str) -> None:
    """Create an identifier file in the collection directory.

    Args:
        path: Path to the collection directory
        uid: UID of the collection
    """
    path = Path(path)
    with open(path.joinpath(_identifier_filename(uid)), "w") as f:
        f.write("Date of creation: " + str(datetime.now()))


def _identifier_filename(uid: str) -> str:
    return IDENTIFIER_PREFIX + IDENTIFIER_SEPARATOR + uid


def _validate_path(path: Path, uid: str) -> bool:
    return path.is_dir() and path.joinpath(_identifier_filename(uid)).is_file()


def _find_uid_from_path(path: Path) -> Optional[str]:
    try:
        return path.glob(f"{IDENTIFIER_PREFIX}*").__next__().name.split("-")[1]
    except StopIteration:
        return None


def _find_collection(uid: str, root_dir: Path) -> tuple[Path, ...]:
    """Find the collection with UID under given root_dir.

    Args:
        uid: UID to search for
        root_dir: root directory for search
    """
    try:
        return tuple(
            Path(i).parent.absolute()
            for i in _find_posix(_identifier_filename(uid), root_dir.as_posix())
        )
    except subprocess.CalledProcessError:
        raise NotImplementedError(
            "Only POSIX systems are supported for now. Install `find`."
        )


def _find_posix(iname: str, root_dir: str) -> tuple[str, ...]:
    """Find function using system `find` on linux."""
    # assert that "find" is available
    assert (
        subprocess.run(["which", "find"], capture_output=True).returncode == 0
    ), "command `find` not available"

    completed_process = subprocess.run(
        [
            "find",
            root_dir,
            "-iname",
            iname,
            "-not",
            "-path",
            r"*/\.git/*",
        ],
        capture_output=True,
    )
    identifier_files_found = tuple(
        completed_process.stdout.decode("utf-8").splitlines()
    )
    return identifier_files_found


def _scan_directory_for_collections(root_dir: Path) -> tuple[tuple[str, Path], ...]:
    """Scan the directory for collections.

    Args:
        root_dir: Directory to scan for collections

    Returns:
        Tuple of tuples with the UID and path of the collection
    """

    log.info(f"Scanning {root_dir}")

    if not root_dir.exists():
        log.warning(f"Path does not exist: {root_dir}")
        return ()

    found_indicator_files = _find_posix(
        f"{IDENTIFIER_PREFIX}{IDENTIFIER_SEPARATOR}*", root_dir.as_posix()
    )

    if not found_indicator_files:
        log.info(f"No collections found in {root_dir}")
        return ()

    return tuple(
        (i.split(IDENTIFIER_SEPARATOR)[-1], Path(i).parent)
        for i in found_indicator_files
    )
