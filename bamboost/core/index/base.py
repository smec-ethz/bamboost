# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
"""Module to manage the database index and its ID's.

Attributes:
    THREAD_SAFE: if True, the index is thread safe
    CONVERT_ARRAYS: if True, convert numpy arrays to lists
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path

from bamboost import BAMBOOST_LOGGER

__all__ = [
    "Database",
    "CollectionTable",
    "Entry",
    "find",
    "get_uid_from_path",
    "get_known_paths",
    "uid2",
    "DatabaseNotFoundError",
    "THREAD_SAFE",
    "CONVERT_ARRAYS",
    "PREFIX",
    "DOT_REPLACEMENT",
    "MPI",
]

import os
import subprocess
from dataclasses import dataclass
from time import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import h5py
import pandas as pd

import bamboost.core.index.sqlite_database as sql
from bamboost import config
from bamboost.core.hdf5.file_handler import open_h5file
from bamboost.core.mpi import MPI

if TYPE_CHECKING:
    from mpi4py import MPI

log = BAMBOOST_LOGGER.getChild("database")


PREFIX = ".BAMBOOST-"
"prefix for databaseID identifier file"

DOT_REPLACEMENT = "DOT"
"replace dots with this in column names for sqlite"

_comm = MPI.COMM_WORLD  # TODO: is this good practice?

THREAD_SAFE = False
CONVERT_ARRAYS = True


# ------------------
# Exceptions
# ------------------
class DatabaseNotFoundError(Exception):
    """Exception raised when a database is not found in the index."""

    pass


# ------------------
# Classes
# ------------------

Error = sql.sqlite3.Error
"""Error exception for index errors."""


# Define a TypeVar for a bound method of IndexAPI
IndexAPIMethod = TypeVar("IndexAPIMethod", bound=Callable[..., Any])


class DatabaseProtocol(Protocol):
    engine: sql.SQLEngine


def get(method: IndexAPIMethod) -> IndexAPIMethod:
    """Decorator for methods in IndexAPI to handle MPI broadcasting.

    Args:
        method: A method of IndexAPI, where the first argument is `self`.

    Returns:
        A decorated function that takes the same arguments as `method`.
    """
    comm = MPI.COMM_WORLD

    @wraps(method)
    def inner(self: DatabaseProtocol, *args, **kwargs):
        # handle MPI
        if comm.rank == 0:
            with self.engine.open():
                result = method(self, *args, **kwargs)

        return comm.bcast(result, root=0)

    return cast(IndexAPIMethod, inner)


def write(method: IndexAPIMethod) -> IndexAPIMethod:
    comm = MPI.COMM_WORLD

    @wraps(method)
    def inner_root(self: Database, *args, **kwargs):
        with self.engine.open():
            method(self, *args, **kwargs)

    def inner_off_root(*_args, **_kwargs):
        pass

    if comm.rank == 0:
        return cast(IndexAPIMethod, inner_root)
    else:
        return cast(IndexAPIMethod, inner_off_root)


class Database:
    """Main class to manage the database of bamboost collections."""

    COLLECTIONS_TABLE: str = "collections"

    def __init__(
        self,
        file: str | Path = config.index.databaseFile,
    ):
        self.engine = sql.SQLEngine(file)
        self.create_collections_table()
        self.clean()
        self._initialized = True

    def __repr__(self) -> str:
        return self.read_table().__repr__()

    def _repr_html_(self) -> str | None:
        return self.read_table()._repr_html_()

    @get
    def __getitem__(self, id: str) -> CollectionTable:
        return CollectionTable(id, database=self)

    def _ipython_key_completions_(self) -> list:
        return self.read_table().id.tolist()

    @write
    def create_collections_table(self) -> None:
        """Create the index table if it does not exist."""
        self.engine.cursor.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.COLLECTIONS_TABLE} (id TEXT PRIMARY KEY, path TEXT)"""
        )

    def get_collection_table(self, id: str) -> CollectionTable:
        """Get the table of a database.

        Args:
            - id (str): ID of the database
        """
        return CollectionTable(id, database=self)

    @get
    def read_table(self, *args, **kwargs) -> pd.DataFrame:
        """Read the index table.

        Returns:
            pd.DataFrame: index table
        """
        return pd.read_sql_query(
            f"SELECT * FROM {self.COLLECTIONS_TABLE}", self.engine.conn, *args, **kwargs
        )

    @get
    def fetch(self, query: str, *args, **kwargs) -> list[Any]:
        """Query the index table.

        Args:
            query (str): query string
        """
        return self.engine.execute(query, *args, **kwargs).fetchall()

    @get
    def get_path(self, id: str) -> str:
        """Get the path of a database from its ID.

        Args:
            id (str): ID of the database

        Returns:
            str: path of the database
        """
        path = self.engine.execute(
            f"SELECT path FROM {self.COLLECTIONS_TABLE} WHERE id=?", (id,)
        ).fetchone()
        if path and _check_path(id, path[0]):
            return path[0]

        # if path is wrong, try to find it
        for root_dir in config.index.searchPaths:
            log.debug(f"Searching for database {id} in {root_dir}")
            res = find(id, root_dir)
            if res:
                path = os.path.dirname(res[0])
                self.insert_path(id, path)
                return path

        raise FileNotFoundError(f"Database {id} not found on system.")

    @write
    def insert_path(self, id: str, path: str) -> None:
        """Insert a database path into the index.

        Args:
            id: ID of the database
            path: path of the database
        """
        path = os.path.abspath(path)
        self.engine.execute(
            f"INSERT OR REPLACE INTO {self.COLLECTIONS_TABLE} VALUES (?, ?)", (id, path)
        )

    @get
    def get_id(self, path: str) -> str:
        """Get the ID of a database from its path.

        Args:
            path (str): path of the database

        Returns:
            str: ID of the database

        Raises:
            DatabaseNotFoundError: if the database is not found in the index
        """
        path = os.path.abspath(path)
        found_id = self.engine.execute(
            "SELECT id FROM dbindex WHERE path=?", (path,)
        ).fetchone()
        if found_id is None:
            raise DatabaseNotFoundError(f"Database at {path} not found in index.")
        return found_id[0]

    @write
    def scan_known_paths(
        self, search_paths: Sequence[Union[str, Path]] = config.index.searchPaths
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

            databases_found = completed_process.stdout.splitlines()

            if not databases_found:
                log.info(f"No databases found in {path}")
                continue

            for database in databases_found:
                name = Path(database).name
                try:
                    id = name.split("-")[1]
                    self.insert_path(id, Path(database).parent.as_posix())
                except IndexError:
                    log.warning(f"Invalid database name format: {name}")

    @write
    def clean(self, purge: bool = False) -> Database:
        """Clean the index from wrong paths.

        Args:
            purge: Also deletes the table of unmatching
                uid/path pairs. Defaults to False.
        """
        collections = self.engine.execute(
            f"SELECT id, path FROM {self.COLLECTIONS_TABLE}"
        ).fetchall()
        for id, path in collections:
            if not _check_path(id, path):
                self.drop_path(id)

        if purge:
            # all tables starting with db_ are tables of databases
            all_tables = self.engine.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            id_list_tables = {
                i[0].split("_")[1] for i in all_tables if i[0].startswith("db_")
            }
            id_list = self.engine.execute("SELECT id FROM dbindex").fetchall()

            for id in id_list_tables:
                if id not in id_list:
                    self.engine.execute(f"DROP TABLE db_{id}")
                    self.engine.execute(f"DROP TABLE db_{id}_t")
                    log.debug(f"Removed table db_{id}")

        return self

    @write
    def drop_path(self, id: str) -> None:
        """Drop a path from the index.

        Args:
            id: ID of the database
        """
        self.engine.execute(f"DELETE FROM {self.COLLECTIONS_TABLE} WHERE id=?", (id,))
        log.debug(f"Removed {id} from collections table")

    def check_path(self, id: str, path: str) -> bool:
        """Check if path is going to the correct database."""
        return _check_path(id, path)


class CollectionTable:
    """Class to manage the table of a database. Multiton pattern. One table per
    database.
    """

    def __init__(
        self,
        id: str,
        database: Database,
        engine: sql.SQLEngine | None = None,
    ):
        self.id: str = id
        self.database: Database = database
        self.engine: sql.SQLEngine = engine or database.engine
        self.path: str = self.database.get_path(self.id)

        self.TABLENAME = f"coll_{self.id}"
        self.TABLENAME_UPDATE_TIME = f"coll_{self.id}_t"

        self._entries = {}
        self._initialized = True
        self.create_database_table()

    @get
    def read_table(self) -> pd.DataFrame:
        """Read the table of the database.

        Returns:
            Table of the database
        """
        df = pd.read_sql_query(f"SELECT * FROM {self.TABLENAME}", self.engine.conn)
        # drop "hidden" columns which start with _
        df = df.loc[:, ~df.columns.str.startswith("_")]
        df.rename(columns=lambda x: x.replace(DOT_REPLACEMENT, "."), inplace=True)
        return df

    @get
    def read_entry(self, entry_id: str) -> pd.Series:
        """Read an entry from the database.

        Args:
            entry_id (str): ID of the entry

        Returns:
            pd.Series: entry from the database
        """
        cursor = self.engine.execute(
            f"SELECT * FROM {self.TABLENAME} WHERE id=?", (entry_id,)
        )
        series = pd.Series(*cursor.fetchone())
        print(series)
        series.index = [description[0] for description in cursor.description]
        print(cursor.description)
        series.rename(index=lambda x: x.replace(DOT_REPLACEMENT, "."), inplace=True)
        return series

    @get
    def read_column(self, *columns: str) -> pd.DataFrame:
        """Read columns from the database.

        Args:
            *columns (list): columns to read

        Returns:
            pd.DataFrame: columns from the database
        """
        cursor = self.engine.execute(
            f"SELECT {', '.join(columns)} FROM {self.TABLENAME}"
        )
        df = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
        df.rename(columns=lambda x: x.replace(DOT_REPLACEMENT, "."), inplace=True)
        return df

    @write
    def drop_table(self) -> None:
        """Drop the table of the database."""
        self.engine.execute(f"DROP TABLE {self.TABLENAME}")
        self.engine.execute(f"DROP TABLE {self.TABLENAME_UPDATE_TIME}")

    @write
    def create_database_table(self) -> None:
        """Create a table for a database."""
        self.engine.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.TABLENAME} 
                (id TEXT PRIMARY KEY NOT NULL, time_stamp DATETIME, notes TEXT, processors INTEGER)
            """
        )
        self.engine.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.TABLENAME_UPDATE_TIME} (id TEXT PRIMARY KEY,
                update_time DATETIME)
            """
        )

    @write
    def update_entry(self, entry_id: str, data: dict) -> None:
        """Update an entry in the database.

        Args:
            entry_id (str): ID of the entry
            data (dict): data to update
        """
        # return if data is empty
        if not data:
            return

        # get columns of table
        cursor = self.engine.execute(f"PRAGMA table_info({self.TABLENAME})")
        cols = cursor.fetchall()

        # replace dots in keys
        for key in list(data.keys()):
            new_key = key.replace(".", DOT_REPLACEMENT)
            new_key = _remove_illegal_column_characters(new_key)
            if new_key != key:
                data[new_key] = data.pop(key)

        # check if columns exist
        for key, val in data.items():
            # key = key.replace(".", DOT_REPLACEMENT)
            # key = _remove_illegal_column_characters(key)
            if any(key == column[1] for column in cols):
                continue
            dtype = sql.get_sqlite_column_type(val)
            self.engine.execute(
                f"ALTER TABLE {self.TABLENAME} ADD COLUMN [{key}] {dtype}"
            )

        # insert data into table
        data.pop("id", None)

        keys = ", ".join([f"[{key}]" for key in data.keys()])
        values = ", ".join([f":{key}" for key in data.keys()])
        updates = ", ".join([f"[{key}] = excluded.[{key}]" for key in data.keys()])

        query = f"""
        INSERT INTO {self.TABLENAME} (id, {keys})
        VALUES (:id, {values})
        ON CONFLICT(id) DO UPDATE SET
        {updates}
        """
        data["id"] = entry_id
        self.engine.execute(query, data)

        # update update time
        self.engine.execute(
            f"INSERT OR REPLACE INTO {self.TABLENAME_UPDATE_TIME} VALUES (?, ?)",
            (entry_id, time()),
        )

    @write
    def sync(self) -> None:
        """Sync the table with the file system."""
        all_ids_fs = set(
            [
                i
                for i in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, i))
            ]
        )

        cursor = self.engine.execute(
            f"SELECT id, update_time FROM {self.TABLENAME_UPDATE_TIME}"
        )

        for id, last_up_time in cursor.fetchall():
            # remove entries that do not exist on the file system
            if id not in all_ids_fs:
                self.engine.execute(f"DELETE FROM {self.TABLENAME} WHERE id=?", (id,))
                self.engine.execute(
                    f"DELETE FROM {self.TABLENAME_UPDATE_TIME} WHERE id=?", (id,)
                )
                continue

            # update entries that have been modified
            all_ids_fs.remove(id)
            if self.entry(id).mtime > last_up_time:
                self.update_entry(id, self.entry(id).get_all_metadata())

        # add new entries
        for id in all_ids_fs:
            self.update_entry(id, self.entry(id).get_all_metadata())

    @get
    def entry(self, entry_id: str) -> Entry:
        """Get the Entry object of an entry.
        Multiton pattern. One Entry per entry.
        """
        if entry_id not in self._entries:
            self._entries[entry_id] = Entry(entry_id, self.path)
        return self._entries[entry_id]


@dataclass
class Entry:
    """Simulation entry in a database.
    Simplified version of the Simulation class in the simulation module.
    """

    def __init__(self, id: str, path: str) -> None:
        self.id = id
        self.path = path
        self.h5file = os.path.join(self.path, self.id, f"{self.id}.h5")

    @property
    def metadata(self) -> dict:
        """Get the metadata of the entry."""
        with open_h5file(self.h5file, "r") as file:
            return dict(file.attrs)

    @property
    def parameters(self) -> dict:
        """Get the parameters of the entry."""
        tmp_dict = dict()
        with open_h5file(self.h5file, "r") as file:
            try:
                tmp_dict.update(file["parameters"].attrs)
                for key in cast(h5py.Group, file["parameters"]).keys():
                    tmp_dict.update(
                        {key: cast(h5py.Dataset, file[f"parameters/{key}"])[()]}
                    )
            except KeyError:
                pass

        return tmp_dict

    def get_all_metadata(self) -> dict:
        """Get all metadata of the entry."""
        return {**self.metadata, **self.parameters}

    @property
    def mtime(self) -> float:
        """Get the modification time of the entry."""
        return os.path.getmtime(self.h5file)


# ----------------------
# Module level functions
# ----------------------


def find(uid, root_dir) -> list:
    """Find the database with UID under given root_dir.

    Args:
        uid: UID to search for
        root_dir: root directory for search
    """
    if os.name == "posix":
        paths = _find_posix(uid, root_dir)
    else:
        paths = _find_python(uid, root_dir)
    if len(paths) > 1:
        log.warning(f"Multiple paths found for UID {uid}:\n{paths}")
    return paths


def get_uid_from_path(path: str) -> str:
    """Returns the UID found in the specified path."""
    for file in os.listdir(path):
        if file.startswith(".BAMBOOST"):
            return file.split("-")[1]
    raise FileNotFoundError("No UID file found at specified path.")


def get_known_paths() -> list:
    return config.index.searchPaths


def _find_posix(uid, root_dir) -> list:
    """Find function using system `find` on linux."""
    completed_process = subprocess.run(
        ["find", root_dir, "-iname", uid2(uid), "-not", "-path", r"*/\.git/*"],
        capture_output=True,
    )
    paths_found = completed_process.stdout.decode("utf-8").splitlines()
    return paths_found


def _find_python(uid, root_dir) -> None:
    """Some find function for Windows or other if `find` is not working.

    TODO: to be implemented
    """
    pass


def uid2(uid) -> str:
    return f"{PREFIX}{uid}"


def _check_path(uid: str, path: str) -> bool:
    """Check if path is going to the correct database"""
    if not os.path.exists(path):
        return False
    if f"{PREFIX}{uid}" in os.listdir(path):
        return True
    return False


def _remove_illegal_column_characters(key: str) -> str:
    """Remove illegal characters in sqlite column names from a string.

    Args:
        key (str): key to clean

    Removes all content in parenthesis and replaces dashes with underscores.
    """
    import re

    # clean from parenthesis
    key = re.sub(r"[()]", "", re.sub(r"\(.*?\)", "", key))
    # clean from dashes
    key = re.sub(r"-", "_", key)
    return key
