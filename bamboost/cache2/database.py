from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Tuple, Union

from sqlalchemy.exc import NoResultFound
from typing_extensions import TypeAlias

from bamboost import BAMBOOST_LOGGER, config
from bamboost.cache2.model import (
    CacheAPI,
    Collection,
    _SimulationMetadataT,
    _SimulationParameterT,
)

log = BAMBOOST_LOGGER.getChild("Database")

PREFIX = ".BAMBOOST-"


StrPath: TypeAlias = Union[str, Path]


class Database:
    def __init__(self, cache: Optional[CacheAPI] = None) -> None:
        self._cache = cache or CacheAPI(config.index.databaseFile)

    def scan_for_collections(
        self,
        *,
        search_paths: Set[Path] = config.index.searchPaths,
    ) -> None:
        """Scan known paths for databases and update the index.

        Args:
            search_paths (List[Path], optional): Paths to scan for databases.
                Defaults to config.index.searchPaths.
        """
        for path in search_paths:
            log.info(f"Scanning {path}")

            if not path.exists():
                log.warning(f"Path does not exist: {path}")
                continue

            try:
                completed_process = subprocess.run(
                    [
                        "find",
                        path.as_posix(),
                        "-iname",
                        f"{PREFIX}*",
                        "-not",
                        "-path",
                        "*/.git/*",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,  # Raise exception if the command fails
                )
            except subprocess.CalledProcessError as e:
                log.error(f"Error scanning path {path}: {e}")
                continue

            found_collections = completed_process.stdout.splitlines()

            if not found_collections:
                log.info(f"No collections found in {path}")
                continue

            with self._cache.scoped_session(make_changes=True):
                for collection in found_collections:
                    log.info(f"Found collection: {collection}")
                    identifier_file = Path(collection)
                    self._cache.update_collection(
                        uid=identifier_file.name.split("-")[1],
                        path=identifier_file.parent.as_posix(),
                    )

    def check_integrity(self) -> None:
        """Check the integrity of the cache.

        This method checks if the paths stored in the cache are valid. If a
        path is not valid, it is removed from the cache.
        """
        for coll in self._cache.get_collections():
            if not _validate_path(Path(coll.path), coll.uid):
                log.warning(f"Invalid path in cache: {coll.path}")
                self._cache.delete(coll)

    def resolve_path(
        self,
        uid: str,
        *,
        search_paths: Set[Path] = config.index.searchPaths,
    ) -> Path:
        try:
            stored_path = self._cache.get_collection(uid).path
            stored_path = Path(stored_path)
            if _validate_path(stored_path, uid):
                return stored_path
            else:
                log.debug(f"--> Found path in cache for collection {uid} is not valid.")
        except NoResultFound:
            log.debug(f"--> No path found in cache for collection {uid}")

        # Try to find the database in the search paths
        for root_dir in search_paths:
            log.debug(f"--> Searching for collection {uid} in {root_dir}")
            paths_found = _find_collection(uid, Path(root_dir))
            if len(paths_found) > 0:  # If at least one file is found
                if len(paths_found) > 1:
                    log.warning(
                        f"Multiple collections found for {uid}. Using the first one."
                        f"\n{paths_found}"
                    )
                self._cache.update_collection(uid, path=paths_found[0].as_posix())
                return paths_found[0]

        raise FileNotFoundError(f"Database with {uid} was not found.")

    def resolve_uid(self, path: str | Path) -> Optional[str]:
        """Resolve the UID of a collection from a path.

        Returns the UID of the collection or `None` if it can't be determined.

        Args:
            path: Path of the collection
        """
        path = Path(path)
        cached_uid = self._cache.get_collection(path=path.as_posix()).id
        if _validate_path(path, cached_uid):
            return cached_uid

        log.debug(f"--> Found path in cache for collection {cached_uid} is not valid.")
        uid = _find_uid_from_path(path)
        return uid

    def update_collection(self, uid: str, path: StrPath) -> None:
        path = Path(path)
        self._cache.update_collection(uid, path.as_posix())

    def sync_collection(self, uid: str, path: Optional[StrPath] = None) -> None:
        """Sync the table with the file system."""
        if path is None:
            path = self.resolve_path(uid)
        else:
            path = Path(path)

        all_entries_fs = set((i.name for i in path.iterdir() if i.is_dir()))

        with self._cache.scoped_session(make_changes=True):
            for sim in self._cache.get_collection(uid).simulations:
                if sim.name not in all_entries_fs:
                    self._cache.delete(sim)
                    continue

                all_entries_fs.remove(sim.name)
                h5_file = path.joinpath(sim.name, f"{sim.name}.h5")

                if (  # type: ignore
                    datetime.fromtimestamp(h5_file.stat().st_mtime) > sim.modified_at
                ):
                    metadata, params = simulation_metadata_from_h5(h5_file)
                    sim.update_metadata(metadata)
                    sim.update_parameters(params)

        for name in all_entries_fs:
            self.cache_simulation(collection_id=uid, simulation_name=name)

    def drop_collection(self, uid: str) -> None:
        self._cache.delete_collection(uid)

    def drop_simulation(self, collection_id: str, simulation_name: str) -> None:
        self._cache.delete_simulation(collection_id, simulation_name)

    def cache_simulation(
        self,
        collection_id: str,
        simulation_name: str,
        *,
        collection_path: Optional[StrPath] = None,
    ) -> None:
        if collection_path is None:
            collection_path = self.resolve_path(collection_id)
        else:
            collection_path = Path(collection_path)

        h5_file = collection_path.joinpath(simulation_name, f"{simulation_name}.h5")
        metadata, params = simulation_metadata_from_h5(h5_file)
        self._cache.update_simulation(collection_id, simulation_name, metadata, params)

    def get_collection(self, uid: str) -> Collection:
        return self._cache.get_collection(uid)

    def get_collections(self) -> list[Collection]:
        return self._cache.get_collections()

    def get_simulation(self, collection_id: str, simulation_name: str):
        return self._cache.get_simulation(collection_id, simulation_name)


def _identifier_filename(uid: str) -> str:
    return PREFIX + uid


def _validate_path(path: Path, uid: str) -> bool:
    return path.is_dir() and path.joinpath(_identifier_filename(uid)).is_file()


def _find_uid_from_path(path: StrPath) -> Optional[str]:
    path = Path(path)
    try:
        return path.glob(f"{PREFIX}*").__next__().name.split("-")[1]
    except StopIteration:
        return None


def _find_collection(uid: str, root_dir: Path) -> tuple[Path, ...]:
    """Find the database with UID under given root_dir.

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


def simulation_metadata_from_h5(
    file: Path,
) -> Tuple[_SimulationMetadataT, _SimulationParameterT]:
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
