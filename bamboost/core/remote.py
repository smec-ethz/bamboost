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
from sqlalchemy.orm import sessionmaker

from bamboost import BAMBOOST_LOGGER, _config, constants
from bamboost._config import config
from bamboost.core.collection import Collection, _FilterKeys
from bamboost.core.simulation.base import Simulation
from bamboost.index._filtering import Filter, Sorter
from bamboost.index.base import (
    CollectionUID,
    Index,
    create_identifier_file,
    get_identifier_filename,
)
from bamboost.index.store import json_deserializer, json_serializer
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

    DATABASE_REMOTE_PATH = Path(_config._LOCAL_DIR).joinpath(
        constants.DEFAULT_DATABASE_FILE_NAME
    )
    WORKSPACE_SPLITTER = "_WS_"

    @staticmethod
    def _make_local_db_name(id: str, version: str | None = "0.11") -> str:
        return f"{id}.{version}.sqlite"

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
                ".bamboost_cache", constants.DEFAULT_DATABASE_FILE_NAME
            )

        # Create the local path if it doesn't exist
        self._local_path.mkdir(parents=True, exist_ok=True)
        self._local_database_path = cache_dir.joinpath(
            self._make_local_db_name(self.id, "0.11")
        )

        # super init
        self.search_paths = PathSet([self._local_path])
        self._url = f"sqlite:///{self._local_database_path}"

        # Fetch the remote database
        if not skip_fetch:
            # this is a blocking call to rsync
            self.fetch_remote_database()

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

    def _fetch_remote_database(
        self, db_name: str, destination: StrPath | None = None
    ) -> subprocess.Popen:
        """Fetch the remote SQL database using rsync.

        Args:
            db_name (str): The name of the database file to fetch.
            destination (StrPath): The local destination path for the database file.

        Returns:
            subprocess.Popen: The Popen object for the rsync process.
        """
        return subprocess.Popen(
            [
                "rsync",
                "-av",
                f"{self._remote_url}:{Path(_config._LOCAL_DIR).joinpath(db_name)}",
                str(destination or self._local_database_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def fetch_remote_database(
        self,
        *,
        migrate_from: str | None = None,
    ) -> None:
        """Fetch the remote SQL database."""

        if migrate_from is not None:
            from bamboost.index.versioning import Version, migrate_database

            source_version = Version.from_str(migrate_from)
            # raises ValueError if invalid

            cache_dir = Path(config.paths.cacheDir)
            tmp_dir = cache_dir.joinpath(".tmp")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_db_path = tmp_dir.joinpath(
                self._make_local_db_name(self.id, migrate_from)
            )
            stream_popen_output(self._fetch_remote_database)(
                source_version.database_file_name, tmp_db_path
            )

            migrate_database(
                source_version,
                Version.latest(),
                source_db=tmp_db_path,
                destination_db=self._local_database_path,
                update=True,
            )

        else:
            stream_popen_output(self._fetch_remote_database)(
                str(self._remote_database_path)
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
    _index: Remote

    def __init__(
        self,
        uid: str,
        remote: Remote,
        *,
        filter: Filter | None = None,
        sorter: Sorter | None = None,
    ):
        self.uid = CollectionUID(uid)
        self._index = remote

        # A key store with completion of all the parameters and metadata keys
        self.k = _FilterKeys(self)

        # Resolve the path (this updates the index if necessary)
        self.path = remote._local_path.joinpath(uid)
        self.remote_path = cast(Path, remote._get_collection_path(self.uid))

        # Create the diretory for the collection if necessary
        self.path.mkdir(parents=True, exist_ok=True)

        # Check if identifier file exists (and create it if necessary)
        if not self.path.joinpath(get_identifier_filename(uid=self.uid)).exists():
            create_identifier_file(self.path, self.uid)

        self._filter = filter
        self._sorter = sorter

    def _repr_html_(self) -> str:
        """HTML repr for ipython/notebooks, using jinja2 for templating."""
        import pkgutil

        from jinja2 import Template

        html_string = pkgutil.get_data("bamboost", "_repr/manager.html").decode()  # type: ignore
        icon = pkgutil.get_data("bamboost", "_repr/icon.txt").decode()  # type: ignore

        template = Template(html_string)

        return template.render(
            icon=icon,
            db_path=f"<a href={self.path.as_posix()}>{self.path}</a>",
            db_uid=f"{self.uid} ({self._index._remote_url})",
            db_size=len(self),
            _filter=self._filter,
            _sort=self._sorter,
        )

    def __getitem__(self, name_or_idx: str | int) -> RemoteSimulation:
        if isinstance(name_or_idx, int):
            name: str = self.df.iloc[name_or_idx]["name"]
        else:
            name = name_or_idx
        return RemoteSimulation(
            name, self.path, collection_uid=self.uid, index=self._index, comm=self._comm
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
            # source will be past to rsync as <remote_url>:<source>
            # source should be the glob: <remote_path>/*
            source = self.remote_path.joinpath("*")
            dest = self.path

        stream_popen_output(self._index.rsync)(source, dest)
        return self


class RemoteSimulation(Simulation):
    _index: Remote
    remote_path: Path

    def __init__(
        self,
        name: str,
        parent: StrPath,
        collection_uid: CollectionUID,
        index: Remote,
        comm: Optional[Comm] = None,
        **kwargs,
    ):
        # if there is no local path, sync it from remote
        # assumes that this is the first time accessing the simulation
        # initialization of base Simulation requires the path to exist
        # (including the data.h5 file)
        path = index.get_local_path(collection_uid).joinpath(name)
        self.remote_path: Path = index._get_collection_path(collection_uid).joinpath(
            name
        )
        if not path.exists() or not path.joinpath("data.h5").exists():
            log.info(
                f"Local simulation '{name}' not found in collection '{collection_uid}'. "
                "Syncing from remote..."
            )
            stream_popen_output(index.rsync)(self.remote_path, path.parent)

        # call super init which requires the path to exist otherwise it will raise FileNotFoundError
        super().__init__(
            name,
            parent,
            index=index,
            comm=comm,
            collection_uid=collection_uid,
            **kwargs,
        )

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
