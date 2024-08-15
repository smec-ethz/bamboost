# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
"""Module to manage the database index and its ID's."""

from __future__ import annotations

from bamboost import BAMBOOST_LOGGER

__all__ = [
    "Null",
    "IndexAPI",
    "DatabaseTable",
    "Entry",
    "find",
    "get_uid_from_path",
    "get_known_paths",
]

import json
import os
import subprocess
from dataclasses import dataclass
from time import time
from typing import Callable, Generator, Iterable

import pandas as pd

import bamboost._sqlite_database as sql
from bamboost._config import config, paths
from bamboost.common.file_handler import open_h5file
from bamboost.common.mpi import MPI

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


PREFIX = ".BAMBOOST-"  # prefix for databaseID identifier file
DOT_REPLACEMENT = "DOT"  # replace dots with this in column names for sqlite
_comm = MPI.COMM_WORLD  # TODO: is this good practice?


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


class Null:
    """Null object to replace API classes for off-root processes."""

    def __getattr__(self, _):
        return self

    def __bool__(self):
        # Allows the instance to behave like `None` in boolean contexts
        return False

    def __call__(self, *args, **kwargs):
        # Allows the instance to be called like a function
        return self

    def __getitem__(self, _):
        return self

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        return False


class IndexAPI(sql.SQLiteHandler):
    """SQLite database to store database ID, path lookup. As well as the table for
    each database. Location: ~/.local/share/bamboost
    Singleton pattern.

    Attributes:
        - file (str): path to the database file
        - _conn (sqlite3.Connection): connection to the database
        - _cursor (sqlite3.Cursor): cursor for the database
    """

    _instances = {}

    def __new__(cls, *, _file: str = None) -> IndexAPI:
        if _comm.rank != 0:
            return Null()

        _file = _file or paths["DATABASE_FILE"]
        if _file not in cls._instances:
            cls._instances[_file] = super().__new__(cls)
        return cls._instances[_file]

    def __init__(self, *, _file: str = None):
        if hasattr(self, "_initialized"):
            return
        _file = _file or paths["DATABASE_FILE"]
        super().__init__(file=_file)
        self.create_index_table()
        self.clean()
        self._initialized = True

    @classmethod
    def ThreadSafe(cls, *, _file: str = None) -> IndexAPI:
        instance = super().__new__(cls)
        instance.__init__(_file=_file)
        return instance

    def __repr__(self) -> str:
        return self.read_table().__repr__()

    def _repr_html_(self) -> str:
        return self.read_table()._repr_html_()

    def __getitem__(self, id: str) -> DatabaseTable:
        return DatabaseTable(id, _index=self)

    def _ipython_key_completions_(self) -> list:
        return self.read_table().id.tolist()

    @sql.with_connection
    def create_index_table(self) -> None:
        """Create the index table if it does not exist."""
        self._cursor.execute(
            """CREATE TABLE IF NOT EXISTS dbindex (id TEXT PRIMARY KEY, path TEXT)"""
        )

    def get_database_table(self, id: str) -> DatabaseTable:
        """Get the table of a database.

        Args:
            - id (str): ID of the database
        """
        return DatabaseTable(id, _index=self)

    @sql.with_connection
    def read_table(self, *args, **kwargs) -> pd.DataFrame:
        """Read the index table.

        Returns:
            pd.DataFrame: index table
        """
        return pd.read_sql_query("SELECT * FROM dbindex", self._conn, *args, **kwargs)

    @sql.with_connection
    def fetch(self, query: str, *args, **kwargs) -> pd.DataFrame:
        """Query the index table.

        Args:
            query (str): query string
        """
        self._cursor.execute(query, *args, **kwargs)
        return self._cursor.fetchall()

    @sql.with_connection
    def get_path(self, id: str) -> str:
        """Get the path of a database from its ID.

        Args:
            id (str): ID of the database

        Returns:
            str: path of the database
        """
        self._cursor.execute("SELECT path FROM dbindex WHERE id=?", (id,))
        path_db = self._cursor.fetchone()
        if path_db and _check_path(id, path_db[0]):
            return path_db[0]

        # if path is wrong, try to find it
        for root_dir in get_known_paths():
            res = find(id, root_dir)
            if res:
                path = os.path.dirname(res[0])
                self.insert_path(id, path)
                return path

        # last resort, check home
        res = find(id, paths["HOME"])
        if res:
            path = os.path.dirname(res[0])
            self.insert_path(id, path)
            return path

        raise FileNotFoundError(f"Database {id} not found on system.")

    @sql.with_connection
    def insert_path(self, id: str, path: str) -> None:
        """Insert a database path into the index.

        Args:
            id (str): ID of the database
            path (str): path of the database
        """
        path = os.path.abspath(path)
        self._cursor.execute("INSERT OR REPLACE INTO dbindex VALUES (?, ?)", (id, path))

    @sql.with_connection
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
        self._cursor.execute("SELECT id FROM dbindex WHERE path=?", (path,))
        fetched = self._cursor.fetchone()
        if fetched is None:
            raise DatabaseNotFoundError(f"Database at {path} not found in index.")
        return fetched[0]

    @sql.with_connection
    def scan_known_paths(self) -> dict:
        """Scan known paths for databases and update the index."""
        for path in get_known_paths():
            completed_process = subprocess.run(
                ["find", path, "-iname", f"{PREFIX}*", "-not", "-path", r"*/\.git/*"],
                capture_output=True,
            )
            databases_found = completed_process.stdout.decode("utf-8").splitlines()
            for database in databases_found:
                name = os.path.basename(database)
                id = name.split("-")[1]
                self.insert_path(id, os.path.dirname(database))

    def commit_once(self, func) -> Callable:
        """Decorator to bundle changes to a single commit.

        Example:
            >>> @Index.commit_once
            >>> def create_a_bunch_of_simulations():
            >>>     for i in range(1000):
            >>>         db.create_simulation(parameters={...})
            >>>
            >>> create_a_bunch_of_simulations()
        """

        def wrapper(*args, **kwargs):
            with self.open(ensure_commit=True):
                return func(*args, **kwargs)

        return wrapper

    @sql.with_connection
    def clean(self, purge: bool = False) -> IndexAPI:
        """Clean the index from wrong paths.

        Args:
            purge (bool, optional): Also deletes the table of unmatching uid/path pairs. Defaults to False.
        """
        index = self._cursor.execute("SELECT id, path FROM dbindex").fetchall()
        for id, path in index:
            if not _check_path(id, path):
                self._cursor.execute("DELETE FROM dbindex WHERE id=?", (id,))

        if purge:
            # all tables starting with db_ are tables of databases
            all_tables = self._cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            id_list_tables = {
                i[0].split("_")[1] for i in all_tables if i[0].startswith("db_")
            }
            id_list = self._cursor.execute("SELECT id FROM dbindex").fetchall()

            for id in id_list_tables:
                if id not in id_list:
                    self._cursor.execute(f"DROP TABLE db_{id}")
                    self._cursor.execute(f"DROP TABLE db_{id}_t")

        return self

    @sql.with_connection
    def drop_path(self, id: str) -> None:
        """Drop a path from the index.

        Args:
            id (str): ID of the database
        """
        self._cursor.execute("DELETE FROM dbindex WHERE id=?", (id,))

    def check_path(self, id: str, path: str) -> bool:
        """Check if path is going to the correct database."""
        return _check_path(id, path)


class DatabaseTable:
    """
    Class to manage the table of a database. Multiton pattern. One table per
    database.
    """

    _instances = {}

    def __new__(cls, id: str, *args, **kwargs) -> DatabaseTable:
        if _comm.rank != 0:
            return Null()

        if id not in cls._instances:
            cls._instances[id] = super().__new__(cls)
        return cls._instances[id]

    def __init__(self, id: str, *, _index: IndexAPI = None):
        if hasattr(self, "_initialized"):
            return

        self.id = id
        self._entries = {}
        self._initialized = True
        self._index = _index if _index is not None else IndexAPI()
        self.path = self._index.get_path(self.id)
        self.tablename_db = f"db_{self.id}"
        self.tablename_update_times = f"db_{self.id}_t"
        self.create_database_table()

    def __getattr__(self, name):
        if name in {
            "_conn",
            "_cursor",
            "open",
            "close",
            "commit",
            "_is_open",
            "commit_once",
        }:
            return getattr(self._index, name)
        return self.__getattribute__(name)

    # ---------------------
    # Database table functions
    # ---------------------
    @sql.with_connection
    def read_table(self) -> pd.DataFrame:
        """Read the table of the database.

        Returns:
            pd.DataFrame: table of the database
        """
        df = pd.read_sql_query(f"SELECT * FROM {self.tablename_db}", self._conn)
        # drop "hidden" columns which start with _
        df = df.loc[:, ~df.columns.str.startswith("_")]
        df.rename(columns=lambda x: x.replace(DOT_REPLACEMENT, "."), inplace=True)
        return df

    @sql.with_connection
    def read_entry(self, entry_id: str) -> pd.Series:
        """Read an entry from the database.

        Args:
            entry_id (str): ID of the entry

        Returns:
            pd.Series: entry from the database
        """
        self._cursor.execute(
            f"SELECT * FROM {self.tablename_db} WHERE id=?", (entry_id,)
        )
        series = pd.Series(*self._cursor.fetchall())
        series.index = [description[0] for description in self._cursor.description]
        series.rename(index=lambda x: x.replace(DOT_REPLACEMENT, "."), inplace=True)
        return series

    @sql.with_connection
    def read_column(self, *columns: str) -> pd.DataFrame:
        """Read columns from the database.

        Args:
            *columns (list): columns to read

        Returns:
            pd.DataFrame: columns from the database
        """
        self._cursor.execute(f"SELECT {', '.join(columns)} FROM {self.tablename_db}")
        df = pd.DataFrame.from_records(self._cursor.fetchall(), columns=columns)
        df.rename(columns=lambda x: x.replace(DOT_REPLACEMENT, "."), inplace=True)
        return df

    @sql.with_connection
    def drop_table(self) -> None:
        """Drop the table of the database."""
        self._cursor.execute(f"DROP TABLE {self.tablename_db}")
        self._cursor.execute(f"DROP TABLE {self.tablename_update_times}")

    @sql.with_connection
    def create_database_table(self) -> None:
        """Create a table for a database."""
        self._cursor.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.tablename_db} 
                (id TEXT PRIMARY KEY NOT NULL, time_stamp DATETIME, notes TEXT, processors INTEGER)
            """
        )
        self._cursor.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.tablename_update_times} (id TEXT PRIMARY KEY,
                update_time DATETIME)
            """
        )

    @sql.with_connection
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
        self._cursor.execute(f"PRAGMA table_info({self.tablename_db})")
        cols = self._cursor.fetchall()

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
            self._cursor.execute(
                f"ALTER TABLE {self.tablename_db} ADD COLUMN [{key}] {dtype}"
            )

        # insert data into table
        data.pop("id", None)

        keys = ", ".join([f"[{key}]" for key in data.keys()])
        values = ", ".join([f":{key}" for key in data.keys()])
        updates = ", ".join([f"[{key}] = excluded.[{key}]" for key in data.keys()])

        query = f"""
        INSERT INTO {self.tablename_db} (id, {keys})
        VALUES (:id, {values})
        ON CONFLICT(id) DO UPDATE SET
        {updates}
        """
        data["id"] = entry_id
        self._cursor.execute(query, data)

        # update update time
        self._cursor.execute(
            f"INSERT OR REPLACE INTO {self.tablename_update_times} VALUES (?, ?)",
            (entry_id, time()),
        )

    @sql.with_connection
    def sync(self) -> None:
        """Sync the table with the file system."""
        all_ids_fs = set(
            [
                i
                for i in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, i))
            ]
        )

        self._cursor.execute(
            f"SELECT id, update_time FROM {self.tablename_update_times}"
        )

        for id, last_up_time in self._cursor.fetchall():
            # remove entries that do not exist on the file system
            if id not in all_ids_fs:
                self._cursor.execute(
                    f"DELETE FROM {self.tablename_db} WHERE id=?", (id,)
                )
                self._cursor.execute(
                    f"DELETE FROM {self.tablename_update_times} WHERE id=?", (id,)
                )
                continue

            # update entries that have been modified
            all_ids_fs.remove(id)
            if self.entry(id).mtime > last_up_time:
                self.update_entry(id, self.entry(id).get_all_metadata())

        # add new entries
        for id in all_ids_fs:
            self.update_entry(id, self.entry(id).get_all_metadata())

    @sql.with_connection
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
                for key in file["parameters"].keys():
                    tmp_dict.update({key: file[f"parameters/{key}"][()]})
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
    return config["index"].get("paths", [])


def _find_posix(uid, root_dir) -> list:
    """Find function using system `find` on linux."""
    completed_process = subprocess.run(
        ["find", root_dir, "-iname", uid2(uid), "-not", "-path", r"*/\.git/*"],
        capture_output=True,
    )
    paths_found = completed_process.stdout.decode("utf-8").splitlines()
    return paths_found


def _find_python(uid, root_dir) -> list:
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
