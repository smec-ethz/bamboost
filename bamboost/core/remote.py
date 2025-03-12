"""This module introduces the Remote class, which is used to access remote collections."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, cast

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bamboost import BAMBOOST_LOGGER, _config, constants
from bamboost._config import config
from bamboost._typing import Immutable, StrPath
from bamboost.core.collection import Collection
from bamboost.core.simulation.base import Simulation
from bamboost.index.base import (
    CollectionUID,
    Index,
    create_identifier_file,
    get_identifier_filename,
)
from bamboost.index.sqlmodel import json_deserializer, json_serializer
from bamboost.mpi import MPI
from bamboost.utilities import PathSet

if TYPE_CHECKING:
    from pandas import DataFrame
    from typing_extensions import Self

    from bamboost.mpi import Comm

log = BAMBOOST_LOGGER.getChild(__name__)


def capture_popen_output(func: Callable[..., subprocess.Popen]) -> Callable[..., None]:
    """Decorator to capture and print the output of a subprocess.Popen object."""

    def wrapper(*args, **kwargs) -> None:
        process = func(*args, **kwargs)
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
        process.wait()

    return wrapper


class Remote(Index):
    DATABASE_BASE_NAME = "bamboost.sqlite"
    DATABASE_REMOTE_PATH = _config.LOCAL_DIR.joinpath(_config.DATABASE_FILE_NAME)

    def __init__(
        self,
        remote_url: str,
        comm: Optional[Comm] = None,
        *,
        project_path: Optional[str] = None,
        project_name: Optional[str] = None,
        skip_fetch: bool = False,
    ):
        self._remote_url = remote_url
        self._local_path = Path(config.paths.cacheDir).joinpath(remote_url)
        if project_name is not None:
            self._local_path = self._local_path.joinpath(project_name)
        # Create the local path if it doesn't exist
        self._local_path.mkdir(parents=True, exist_ok=True)

        if project_path is not None:
            self._remote_database_path = Path(project_path).joinpath(
                ".bamboost_cache", self.DATABASE_BASE_NAME
            )
        else:
            self._remote_database_path = self.DATABASE_REMOTE_PATH
        self._local_database_path = self._local_path.joinpath(self.DATABASE_BASE_NAME)

        # super init
        self._comm = comm or MPI.COMM_WORLD
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

    def fetch_remote_database(
        self,
    ) -> subprocess.Popen:
        """Fetch the remote database."""
        return subprocess.Popen(
            [
                "rsync",
                "-av",
                f"{self._remote_url}:{self._remote_database_path}",
                str(self._local_database_path),
            ],
            stdout=subprocess.STDOUT,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def rsync(self, source: StrPath, dest: StrPath) -> subprocess.Popen:
        """Sync data with the remote server.

        Args:
            source: The absolute source path on the remote server.
            dest: The relative (from the cache dir for this remote) destination path on
                the local machine.
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
    def list(cls) -> list[str]:
        """List all remote databases in the cache."""
        return [str(name) for name in _config.CACHE_DIR.iterdir() if name.is_dir()]

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
        self._comm = remote._comm
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

    @property
    def df(self) -> DataFrame:
        df = super().df

        # Add a cached column at the second position
        cached_simulations = tuple(str(i) for i in self.path.iterdir() if i.is_dir())
        cached_col = df["name"].isin(cached_simulations)
        df.insert(1, "cached", cached_col)  # pyright: ignore[reportArgumentType]
        return df

    def __getitem__(self, name: str) -> RemoteSimulation:
        return RemoteSimulation(
            name,
            collection_uid=self.uid,
            index=self._index,
            comm=self._comm,
        )

    def _rsync(self, name: Optional[str] = None) -> subprocess.Popen:
        """Transfer data using rsync. This method is called by the `rsync`.
        It returns the subprocess.Popen object.

        Args:
            name: The simulation name of the simulation to be transferred. If None, all
                simulations are synced.
        """
        if name:
            return self._index.rsync(
                self.remote_path.joinpath(name), self.path.joinpath(name)
            )
        else:
            return self._index.rsync(self.remote_path, self.path)

    def rsync(self, name: Optional[str] = None) -> Self:
        """Transfer data using rsync. Wait for the process to finish and return
        self.

        Args:
            name: The simulation name of the simulation to be transferred. If None, all
                simulations are synced.
        """
        capture_popen_output(self._rsync)(name)
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

        # MPI information
        self._comm: Comm = comm or MPI.COMM_WORLD
        self._psize: int = self._comm.size
        self._prank: int = self._comm.rank
        self._ranks = np.array([i for i in range(self._psize)])

        self._data_file: Path = self.path.joinpath(constants.HDF_DATA_FILE_NAME)
        self._xdmf_file: Path = self.path.joinpath(constants.XDMF_FILE_NAME)
        self._bash_file: Path = self.path.joinpath(constants.RUN_FILE_NAME)

    @property
    def parameters(self) -> dict:
        return self._orm.parameter_dict

    @property
    def metadata(self) -> dict:
        return self._orm.as_dict_metadata()

    def rsync(self) -> Self:
        """Sync the simulation data with the remote server."""
        capture_popen_output(self._index.rsync)(self.remote_path, self.path)
        return self
