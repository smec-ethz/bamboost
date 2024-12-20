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
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session, sessionmaker
from typing_extensions import TypeAlias

from bamboost import BAMBOOST_LOGGER, config
from bamboost.core.mpi import MPI, MPISafeMeta, bcast, on_root
from bamboost.index.cache import (
    CacheAPI,
    CollectionORM,
    SimulationORM,
    _json_deserializer,
    _json_serializer,
    _SimulationMetadataT,
    _SimulationParameterT,
    create_all,
)

if TYPE_CHECKING:
    from bamboost.core.mpi import Comm

log = BAMBOOST_LOGGER.getChild("Database")

IDENTIFIER_PREFIX = ".BAMBOOST"
IDENTIFIER_SEPARATOR = "-"

__all__ = [
    "Index",
    "simulation_metadata_from_h5",
    "on_root",
    "CacheAPI",
]

StrPath: TypeAlias = Union[str, Path]


class CollectionUID(str):
    """UID of a collection."""

    def __new__(cls, uid: Optional[str] = None, length: int = 10):
        uid = uid or cls.generate_uid(length)
        return super().__new__(cls, uid.upper())

    @staticmethod
    def generate_uid(length: int) -> str:
        return uuid.uuid4().hex[:length].upper()


class SimulationName(str):
    """Name of a simulation."""

    def __new__(cls, name: Optional[str] = None, length: int = 10):
        name = name or cls.generate_name(length)
        return super().__new__(cls, name)

    @staticmethod
    def generate_name(length: int) -> str:
        return uuid.uuid4().hex[:length]


def with_session(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add a session to the function signature.

    Args:
        func: The function to decorate.
    """

    @wraps(func)
    def inner(self: Index, *args, **kwargs) -> Any:
        if self._context_stack <= 0:
            with cast(Session, self._sm()) as session, session.begin():
                self._s = session
                return func(self, *args, **kwargs)
        else:
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

    _sm: sessionmaker
    _s: Session

    def __init__(
        self, sql_file: Optional[StrPath] = None, comm: Optional[Comm] = None
    ) -> None:
        self._comm = comm or MPI.COMM_WORLD
        self._engine = create_engine(
            f"sqlite:///{sql_file}",
            json_serializer=_json_serializer,
            json_deserializer=_json_deserializer,
        )
        on_root(create_all, self._comm)(self._engine)
        self._sm = sessionmaker(bind=self._engine)

        self._context_stack: int = 0

    def scan_for_collections(
        self,
        *,
        search_paths: Set[Path] = config.index.searchPaths,
    ) -> None:
        """Scan known paths for collections and update the index.

        Iterates through the search paths and searches files with the
        identifier file structure. If a collection is found, it is added to the
        cache.

        Args:
            search_paths (List[Path], optional): Paths to scan for collections.
                Defaults to config.index.searchPaths.
        """
        for path in search_paths:
            found_collections: tuple[tuple[str, Path], ...] = (
                _scan_directory_for_collections(path)
            )
            if not found_collections:
                continue
            collections_data = [
                {"uid": uid, "path": str(path)} for uid, path in found_collections
            ]
            self._s.execute(
                insert(CollectionORM)
                .values(collections_data)
                .on_conflict_do_update(
                    index_elements=["uid"],
                    set_=dict(path=insert(CollectionORM).excluded.path),
                )
            )
            self._s.commit()
            log.info(f"Inserting found collections:\n{found_collections}")

            # for uid, path in found_collections:
            #     log.info(f"Inserting found collection {uid} at {path}")
            #     self._c.update_collection(uid, path.as_posix())

    def check_integrity(self) -> None:
        """Check the integrity of the cache.

        This method checks if the paths stored in the cache are valid. If a
        path is not valid, it is removed from the cache.
        """
        for coll in self._c.get_collections():
            if not _validate_path(Path(coll.path), coll.uid):
                log.warning(f"Invalid path in cache: {coll.path}")
                self._c.delete(coll)

    def resolve_path(
        self,
        uid: str,
        *,
        search_paths: Set[Path] = config.index.searchPaths,
    ) -> Path:
        """Resolve and return the path of a collection from its UID.

        Args:
            uid: UID of the collection
            search_paths: Paths to search for the collection

        Raises:
            FileNotFoundError: If the collection is not found in the search paths
        """
        try:
            stored_path = Path(self._c.get_collection(uid).path)
            if _validate_path(stored_path, uid):
                return stored_path
            else:
                log.debug(f"--> Found path in cache for collection {uid} is not valid.")
        except NoResultFound:
            log.debug(f"--> No path found in cache for collection {uid}")

        # Try to find the collection in the search paths
        for root_dir in search_paths:
            log.debug(f"--> Searching for collection {uid} in {root_dir}")
            paths_found = _find_collection(uid, Path(root_dir))
            if len(paths_found) > 0:  # If at least one file is found
                if len(paths_found) > 1:
                    log.warning(
                        f"Multiple collections found for {uid}. Using the first one."
                        f"\n{paths_found}"
                    )
                self._c.update_collection(uid, path=paths_found[0].as_posix())
                return paths_found[0]

        raise FileNotFoundError(f"Database with {uid} was not found.")

    def resolve_uid(self, path: StrPath) -> Optional[str]:
        """Resolve the UID of a collection from a path.

        Returns the UID of the collection or `None` if it can't be determined.

        Args:
            path: Path of the collection
        """
        path = Path(path)
        cached_uid: str | None = self._s.execute(
            select(CollectionORM.uid).where(CollectionORM.path == path.as_posix())
        ).scalar()
        log.info(f"Cached UID for path {path}: {cached_uid}")
        if cached_uid and _validate_path(path, cached_uid):
            return cached_uid

        log.debug(f"--> Found path in cache for collection {cached_uid} is not valid.")
        uid = _find_uid_from_path(path)
        return uid

    def update_collection(self, uid: str, path: StrPath) -> None:
        """Update the path of a collection in the cache.

        Args:
            uid: UID of the collection
            path: New path of the collection
        """
        path = Path(path)
        self._c.update_collection(uid, path.as_posix())

    def sync_collection(self, uid: str, path: Optional[StrPath] = None) -> None:
        """Sync the table with the file system.

        Iterates through the simulations in the collection and updates the
        metadata and parameters if the HDF5 file has been modified.

        Args:
            uid: UID of the collection
            path (Optional): Path of the collection
        """
        if path is None:
            path = self.resolve_path(uid)
        else:
            path = Path(path)

        all_entries_fs = set((i.name for i in path.iterdir() if i.is_dir()))

        for sim in self.get_collection(uid).simulations:
            if sim.name not in all_entries_fs:
                self._s.delete(sim)
                continue

            all_entries_fs.remove(sim.name)
            h5_file = path.joinpath(sim.name, f"{sim.name}.h5")

            if (  # type: ignore
                datetime.fromtimestamp(h5_file.stat().st_mtime) > sim.modified_at
            ):
                metadata, params = simulation_metadata_from_h5(h5_file)
                self._c.update_simulation(
                    sim.collection_uid, sim.name, metadata, params
                )

        for name in all_entries_fs:
            self.cache_simulation(collection_uid=uid, simulation_name=name)

    def drop_collection(self, uid: str) -> None:
        """Drop a collection from the cache.

        Args:
            uid: UID of the collection
        """
        self._c.delete_collection(uid)

    def drop_simulation(self, collection_uid: str, simulation_name: str) -> None:
        """Drop a simulation from the cache.

        Args:
            collection_uid: UID of the collection
            simulation_name: Name of the simulation
        """
        self._c.delete_simulation(collection_uid, simulation_name)

    def cache_simulation(
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
        if collection_path is None:
            collection_path = self.resolve_path(collection_uid)
        else:
            collection_path = Path(collection_path)

        h5_file = collection_path.joinpath(simulation_name, f"{simulation_name}.h5")
        metadata, params = simulation_metadata_from_h5(h5_file)
        self._c.update_simulation(collection_uid, simulation_name, metadata, params)

    def get_collection(self, uid: str) -> CollectionORM | None:
        return self._s.execute(
            select(CollectionORM).where(CollectionORM.uid == uid)
        ).scalar()

    @bcast
    def get_collections(self) -> Sequence[CollectionORM]:
        return self._s.execute(select(CollectionORM)).scalars().all()

    def get_simulation(
        self, collection_uid: str, simulation_name: str
    ) -> SimulationORM | None:
        # return self._s.get_simulation(collection_uid, simulation_name)
        return self._s.execute(
            select(SimulationORM).where(
                SimulationORM.collection_uid == collection_uid,
                SimulationORM.name == simulation_name,
            )
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
    from bamboost.core.hdf5.file_handler import open_h5file

    with open_h5file(file.as_posix(), "r") as f:
        meta: _SimulationMetadataT = {
            "created_at": datetime.fromisoformat(f.attrs.get("time_stamp", 0)),
            "modified_at": datetime.fromtimestamp(file.stat().st_mtime),
            "description": f.attrs.get("notes", ""),
            "status": f.attrs.get("status", ""),
        }
        params: _SimulationParameterT = dict(f["parameters"].attrs)

        return meta, params


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
            Path(i).parent.absolute() for i in _find_posix(uid, root_dir.as_posix())
        )
    except subprocess.CalledProcessError:
        raise NotImplementedError(
            "Only POSIX systems are supported for now. Install `find`."
        )


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

    try:
        completed_process = subprocess.run(
            [
                "find",
                root_dir.as_posix(),
                "-iname",
                f"{IDENTIFIER_PREFIX}{IDENTIFIER_SEPARATOR}*",
                "-not",
                "-path",
                "*/.git/*",
            ],
            capture_output=True,
            text=True,
            check=True,  # Raise exception if the command fails
        )
    except subprocess.CalledProcessError as e:
        log.error(f"Error scanning path {root_dir}: {e}")
        return ()

    found_indicator_files = completed_process.stdout.splitlines()

    if not found_indicator_files:
        log.info(f"No collections found in {root_dir}")
        return ()

    return tuple(
        (i.split(IDENTIFIER_SEPARATOR)[-1], Path(i).parent)
        for i in found_indicator_files
    )


def _find_posix(uid: str, root_dir: str) -> tuple[str, ...]:
    """Find function using system `find` on linux."""
    completed_process = subprocess.run(
        [
            "find",
            root_dir,
            "-iname",
            _identifier_filename(uid),
            "-not",
            "-path",
            r"*/\.git/*",
        ],
        capture_output=True,
        check=True,
    )
    identifier_files_found = tuple(
        completed_process.stdout.decode("utf-8").splitlines()
    )
    return identifier_files_found
