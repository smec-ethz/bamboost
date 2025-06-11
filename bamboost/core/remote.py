"""Remote Access Module for Bamboost

This module provides the `Remote` class and related classes to facilitate access to remote
collections and simulations in the Bamboost framework.

Classes:
    - Remote: Represents a remote index/database, handles synchronization and workspace management.
    - RemoteCollection: Represents a collection within a remote index, supports data transfer and caching.
    - RemoteSimulation: Represents a simulation within a remote collection, supports remote data access.

Typical usage involves creating a Remote instance pointing to a remote server, listing available
collections, and synchronizing data as needed.

"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, cast

from sqlalchemy import create_engine
from sqlalchemy.orm import remote, sessionmaker

from bamboost import BAMBOOST_LOGGER, _config, constants
from bamboost._config import config
from bamboost.core.collection import Collection
from bamboost.core.simulation.base import Simulation
from bamboost.index.base import (
    CollectionUID,
    Index,
    create_identifier_file,
    get_identifier_filename,
)
from bamboost.index.sqlmodel import json_deserializer, json_serializer
from bamboost.utilities import PathSet

if TYPE_CHECKING:
    from pandas import DataFrame
    from typing_extensions import Self

    from bamboost._typing import _P, StrPath
    from bamboost.mpi import Comm

log = BAMBOOST_LOGGER.getChild(__name__)


def stream_popen_output(func: Callable[_P, subprocess.Popen]) -> Callable[_P, None]:
    """Decorator to await, capture and print the output of a subprocess.Popen object."""

    def wrapper(*args, **kwargs) -> None:
        process = func(*args, **kwargs)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        process.wait()

    return wrapper


class Remote(Index):
    """Represents a remote index/database for accessing and synchronizing collections and simulations.

    The Remote class manages connections to a remote server, handles local caching of remote data,
    and provides methods for synchronizing databases and collections using rsync. It supports
    workspace management and can list available remote databases in the local cache.

    Args:
        remote_url (str): The SSH URL of the remote server.
        comm (Optional[Comm]): Optional MPI communicator.
        workspace_path (Optional[str]): Path to the workspace on the remote server.
        workspace_name (Optional[str]): Name of the workspace.
        skip_fetch (bool): If True, skip fetching the remote database on initialization.

    Attributes:
        DATABASE_BASE_NAME (str): The base name of the database file.
        DATABASE_REMOTE_PATH (Path): Default path to the remote database.
        WORKSPACE_SPLITTER (str): String used to split remote URL and workspace name.
        id (str): Unique identifier for the remote instance.
        _remote_url (str): The SSH URL of the remote server.
        _local_path (Path): Local cache path for the remote data.
        _workspace_path (Optional[str]): Path to the workspace on the remote server.
        _workspace_name (Optional[str]): Name of the workspace.
        _remote_database_path (Path): Path to the remote database file.
        _local_database_path (Path): Path to the local cached database file.
        search_paths (PathSet): Set of search paths for collections.
        _url (str): SQLAlchemy database URL for the local cache.
        _engine: SQLAlchemy engine instance.
        _sm: SQLAlchemy sessionmaker.
        _s: SQLAlchemy session.
    """

    DATABASE_BASE_NAME = "bamboost.sqlite"
    DATABASE_REMOTE_PATH = _config.LOCAL_DIR.joinpath(_config.DATABASE_FILE_NAME)
    WORKSPACE_SPLITTER = "_WS_"

    def __init__(
        self,
        remote_url: str,
        comm: Optional[Comm] = None,
        *,
        workspace_path: Optional[str] = None,
        workspace_name: Optional[str] = None,
        skip_fetch: bool = False,
    ):
        cache_dir = Path(config.paths.cacheDir)
        self._remote_url = remote_url
        self.id = (
            f"{remote_url}{self.WORKSPACE_SPLITTER}{workspace_name}"
            if workspace_name
            else remote_url
        )
        self._local_path = Path(cache_dir).joinpath(self.id)
        self._workspace_path = workspace_path
        self._workspace_name = workspace_name
        self._remote_database_path = self.DATABASE_REMOTE_PATH

        if workspace_name is not None:
            # check if the remote workspace has a meta file
            meta_file = cache_dir.joinpath(f"{self.id}.json")
            if workspace_path is not None:
                # update the meta file
                meta_file.write_text(
                    f'{{"remote_url": "{remote_url}", "workspace_path": "{workspace_path}"}}'
                )
                self._workspace_path = workspace_path
            else:
                assert meta_file.exists(), (
                    f"Workspace {workspace_name} is not found in the cache. "
                    "You must also provide the workspace path (`workspace_path=...`)."
                )
                meta = json.loads(meta_file.read_text())
                self._workspace_path = cast(str, meta["workspace_path"])

            self._remote_database_path = Path(self._workspace_path).joinpath(
                ".bamboost_cache", self.DATABASE_BASE_NAME
            )

        # Create the local path if it doesn't exist
        self._local_path.mkdir(parents=True, exist_ok=True)
        self._local_database_path = cache_dir.joinpath(f"{self.id}.sqlite")

        # super init
        self.search_paths = PathSet([self._local_path])
        self._url = f"sqlite:///{self._local_database_path}"

        # Fetch the remote database
        if not skip_fetch:
            process = self.fetch_remote_database()
            process.wait()

        self._engine = create_engine(
            self._url,
            json_serializer=json_serializer,
            json_deserializer=json_deserializer,
        )
        self._sm = sessionmaker(
            bind=self._engine, autobegin=False, expire_on_commit=False
        )
        self._s = self._sm()

    def __repr__(self) -> str:
        qualname = ".".join([__name__, self.__class__.__qualname__])
        return f"<{qualname} (source={self._remote_url}, workspace={self._workspace_name})>"

    @classmethod
    def _from_id(cls, id: str) -> Remote:
        remote_url, *workspace_name = id.split(cls.WORKSPACE_SPLITTER)
        return cls(
            remote_url,
            workspace_name=workspace_name[0] if workspace_name else None,
            skip_fetch=True,
        )

    def fetch_remote_database(
        self,
    ) -> subprocess.Popen:
        """Fetch the remote SQL database."""
        return subprocess.Popen(
            [
                "rsync",
                "-av",
                f"{self._remote_url}:{self._remote_database_path}",
                str(self._local_database_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def rsync(self, source: StrPath, dest: StrPath) -> subprocess.Popen:
        """Synchronize data from the remote server to the local cache using rsync.

        Args:
            source (StrPath): The absolute source path on the remote server.
            dest (StrPath): The destination path on the local machine, relative
                to the local cache directory for this remote.

        Returns:
            subprocess.Popen: The Popen object for the rsync process.
        """
        return subprocess.Popen(
            [
                "rsync",
                "-ravh",
                f"{self._remote_url}:{source}",
                f"{self._local_path.joinpath(dest)}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    @classmethod
    def list(cls) -> list[Remote]:
        """List all remote databases in the cache."""
        # find all sqlite files in the cache directory
        return [
            Remote._from_id(name.stem)
            for name in _config.CACHE_DIR.iterdir()
            if name.is_file() and name.suffix == ".sqlite"
        ]

    def __getitem__(self, key: str) -> RemoteCollection:
        uid = key.split(" - ")[0]
        return RemoteCollection(uid, remote=self)

    def _ipython_key_completions_(self) -> list[str]:
        return [coll.uid for coll in self._get_collections()]

    def get_local_path(self, collection_uid: str) -> Path:
        return self._local_path.joinpath(collection_uid)


class RemoteCollection(Collection):
    def __init__(
        self,
        uid: str,
        remote: Remote,
    ):
        self.uid = CollectionUID(uid)
        self._index = remote

        # Resolve the path (this updates the index if necessary)
        self.path = remote._local_path.joinpath(uid)
        self.remote_path = cast(Path, remote._get_collection_path(self.uid))

        # Create the diretory for the collection if necessary
        self.path.mkdir(parents=True, exist_ok=True)

        # Check if identifier file exists (and create it if necessary)
        if not self.path.joinpath(get_identifier_filename(uid=self.uid)).exists():
            create_identifier_file(self.path, self.uid)

    def _repr_html_(self) -> str:
        import pkgutil

        html_string = pkgutil.get_data("bamboost", "_repr/manager.html").decode()
        icon = pkgutil.get_data("bamboost", "_repr/icon.txt").decode()
        return (
            html_string.replace("$ICON", icon)
            .replace("$db_path", f"<a href={self.path.as_posix()}>{self.path}</a>")
            .replace("$db_uid", f"{self.uid} [from {self._index._remote_url}]")
            .replace("$db_size", str(len(self)))
        )

    def __getitem__(self, name: str) -> RemoteSimulation:
        return RemoteSimulation(
            name,
            collection_uid=self.uid,
            index=self._index,
            comm=self._comm,
        )

    @property
    def df(self) -> DataFrame:
        df = super().df

        # Add a cached column at the second position
        cached_col = df["name"].isin(self.get_cached_simulation_names())
        df.insert(1, "cached", cached_col)  # pyright: ignore[reportArgumentType]
        return df

    def get_cached_simulation_names(self) -> list[str]:
        return [str(i.name) for i in self.path.iterdir() if i.is_dir()]

    def rsync(self, name: Optional[str] = None) -> Self:
        """Transfer data using rsync. Wait for the process to finish and return
        self.

        Args:
            name: The simulation name of the simulation to be transferred. If None, all
                simulations are synced.
        """
        if name:
            source = self.remote_path.joinpath(name)
            dest = self.path.joinpath(name)
        else:
            source = self.remote_path
            dest = self.path

        stream_popen_output(self._index.rsync)(source, dest)
        return self


class RemoteSimulation(Simulation):
    def __init__(
        self,
        name: str,
        collection_uid: CollectionUID,
        index: Remote,
        comm: Optional[Comm] = None,
        **kwargs,
    ):
        self.name: str = name
        self.path: Path = index.get_local_path(collection_uid).joinpath(name)
        self.remote_path: Path = index._get_collection_path(collection_uid).joinpath(
            name
        )
        self.collection_uid = collection_uid

        # Reference to the database
        self._index: Remote = index

        self._data_file: Path = self.path.joinpath(constants.HDF_DATA_FILE_NAME)
        self._xdmf_file: Path = self.path.joinpath(constants.XDMF_FILE_NAME)
        self._bash_file: Path = self.path.joinpath(constants.RUN_FILE_NAME)

    @property
    def uid(self) -> str:
        return f"ssh://{self._index._remote_url}/{super().uid}"

    @property
    def parameters(self) -> dict:
        return self._orm.parameter_dict

    @property
    def metadata(self) -> dict:
        return self._orm.as_dict_metadata()

    def rsync(self) -> Self:
        """Sync the simulation data with the remote server."""
        stream_popen_output(self._index.rsync)(self.remote_path, self.path.parent)
        return self
