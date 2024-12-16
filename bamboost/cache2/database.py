from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Set, Union

from sqlalchemy import Row
from sqlalchemy.exc import NoResultFound

from bamboost import BAMBOOST_LOGGER, config
from bamboost.cache2.model import CacheAPI

log = BAMBOOST_LOGGER.getChild(__name__)

PREFIX = ".BAMBOOST-"


def _identifier_filename(uid: str) -> str:
    return PREFIX + uid


def _validate_path(path: Path, uid: str) -> bool:
    return path.is_dir() and path.joinpath(_identifier_filename(uid)).is_file()


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


class Database:
    def __init__(self, cache: CacheAPI) -> None:
        self.cache = cache

    def resolve_path(
        self,
        uid: str,
        *,
        search_paths: Set[Path] = config.index.searchPaths,
    ) -> Path:
        try:
            stored_path = self.cache.get_path(uid)
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
                self.cache.insert_path(uid, paths_found[0].as_posix())
                return paths_found[0]

        raise FileNotFoundError(f"Database with {uid} was not found.")

    def resolve_uid(self, path: str | Path) -> str:
        path = Path(path)
        return self.cache.get_uid(path.as_posix())

    def add_collection(self, uid: str, path: str | Path) -> None:
        path = Path(path)
        self.cache.insert_path(uid, path.as_posix())

    def sync_collection(self, uid: str, path: str | Path | None = None) -> None:
        """Sync the table with the file system."""
        if path is None:
            path = self.resolve_path(uid)
        else:
            path = Path(path)

        all_entries_fs = set((i.name for i in path.iterdir() if i.is_dir()))

        for simulation_id, name, modified_at in self.cache.get_from_collection(
            uid, "modified_at"
        ):
            if name not in all_entries_fs:
                self.cache.drop_simulation(simulation_id)
                continue

            all_entries_fs.remove(name)
            if (
                datetime.fromtimestamp(
                    path.joinpath(name, f"{name}.h5").stat().st_mtime
                )
                > modified_at
            ):
                self.cache_simulation(
                    collection_id=uid, simulation_id=simulation_id, simulation_name=name
                )

        for name in all_entries_fs:
            self.cache_simulation(collection_id=uid, simulation_name=name)

    def cache_simulation(
        self,
        collection_id: str,
        simulation_name: str,
        simulation_id: Optional[int] = None,
        *,
        collection_path: Optional[Union[str, Path]] = None,
    ) -> None:
        from bamboost.core.hdf5.file_handler import open_h5file

        if collection_path is None:
            collection_path = self.resolve_path(collection_id)
        else:
            collection_path = Path(collection_path)

        file = collection_path.joinpath(simulation_name, f"{simulation_name}.h5")

        with open_h5file(file.as_posix(), "r") as f:
            self.cache.insert_simulation(
                collection_id=collection_id,
                simulation_name=simulation_name,
                simulation_id=simulation_id,
                created_at=datetime.fromisoformat(f.attrs.get("time_stamp", 0)),
                modified_at=datetime.fromtimestamp(file.stat().st_mtime),
                description=f.attrs.get("notes", ""),
                status=f.attrs.get("status", ""),
                params=dict(f["parameters"].attrs),
            )

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
            path = Path(path)
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

            with self.cache.transaction(make_changes=True):
                for collection in found_collections:
                    log.debug(f"Found collection: {collection}")
                    identifier_file = Path(collection)
                    self.cache.insert_path(
                        identifier_file.name.split("-")[1],
                        identifier_file.parent.as_posix(),
                    )

    def check_integrity(self) -> None:
        """Check the integrity of the cache.

        This method checks if the paths stored in the cache are valid. If a
        path is not valid, it is removed from the cache.
        """
        for uid, path in self.cache.read_all():
            if not _validate_path(Path(path), uid):
                log.warning(f"Invalid path in cache: {path}")
                self.cache.drop_collection(uid)

    def dummy_add_collection(self, uid: str) -> None:
        from bamboost.core.manager import Manager

        _ = Manager(self.resolve_path(uid))

    def get_collection(self, uid: str) -> tuple[Sequence, Sequence[Row[Any]]]:
        return self.cache.get_collection(uid)
