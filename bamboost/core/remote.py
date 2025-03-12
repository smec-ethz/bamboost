"""This module introduces the Remote class, which is used to access remote collections."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bamboost import _config
from bamboost._config import config
from bamboost._typing import StrPath
from bamboost.core.collection import Collection
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
    from bamboost.mpi import Comm


class Remote(Index):
    DATABASE_BASE_NAME = "bamboost.sqlite"
    DATABASE_REMOTE_PATH = _config.LOCAL_DIR.joinpath(_config.DATABASE_FILE_NAME)

    def __init__(
        self,
        remote_url: str,
        comm: Optional[Comm] = None,
        *,
        skip_fetch: bool = False,
    ):
        self._remote_url = remote_url
        self._local_path = Path(config.paths.cacheDir).joinpath(remote_url)
        # Create the local path if it doesn't exist
        self._local_path.mkdir(parents=True, exist_ok=True)
        self._database_path = self._local_path.joinpath(self.DATABASE_BASE_NAME)

        # super init
        self._comm = comm or MPI.COMM_WORLD
        self.search_paths = PathSet([self._local_path])
        self._url = f"sqlite:///{self._database_path}"

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
                f"{self._remote_url}:{self.DATABASE_REMOTE_PATH}",
                str(self._database_path),
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
