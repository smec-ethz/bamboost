# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from functools import wraps
from time import time
from typing import Any, Callable, Generator

import numpy as np
import pandas as pd

from bamboost import index
from bamboost.common.file_handler import open_h5file
from bamboost.common.mpi import MPI

from ..common.sqlite import DATA_TYPES, SYNC_TABLE, parse_sqlite_type

# ----------------
# DECORATORS
# ----------------
def with_connection(func):
    """Decorator to ensure that the cursor is available. If the cursor is not available, the
    connection is opened and closed after the function is executed."""

    @wraps(func)
    def wrapper(self: SQLTable, *args, **kwargs):
        cursor: sqlite3.Cursor = self.cursor
        # check if cursor is available
        if not self._is_open or cursor is None:
            with self.open():
                return func(self, *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


def _on_rank_0(func):
    """Decorator to ensure that the function is only executed on rank 0."""

    @wraps(func)
    def wrapper(self: SQLTable, *args, **kwargs):
        if self._comm.rank == 0:
            return func(self, *args, **kwargs)
        return None

    return wrapper


# ----------------
# CLASSES
# ----------------
class SQLTable:
    """SQLite table for a bamoost database.
    Multiton pattern is used to ensure that only one instance of the class is
    created for each database id.

    Args:
        id (str): The id of the database.
        path (str, optional): The path to the database. Defaults to None.
        _comm (MPI.Comm, optional): The MPI communicator. Defaults to
            MPI.COMM_WORLD.
    """

    _instances = {}
    DOT_REPLACEMENT = "DOT"

    def __new__(cls, id, **kwargs) -> SQLTable:
        if id not in cls._instances:
            cls._instances[id] = super(SQLTable, cls).__new__(cls)
        return cls._instances[id]

    def __init__(
        self,
        id: str,
        *,
        path: str = None,
        _comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        # only rank 0 will interact with the table
        if _comm.rank != 0:
            return
        # only initialize if not already initialized
        if hasattr(self, "_initialized"):
            return

        self.id = id
        self.path = path if path is not None else index.get_path(id)
        self._comm = _comm

        self.file = os.path.abspath(os.path.join(self.path, f".{self.id}.sqlite"))
        self.conn = None
        self.cursor = None
        self._is_open = False
        self._lock = 0

        # self._connect()
        self._entries = dict()
        self._initialized = True

    @property
    def _lock(self) -> int:
        return self._lock_count

    @_lock.setter
    def _lock(self, value: int):
        self._lock_count = value if value >= 0 else 0

    @_on_rank_0
    def _connect(self) -> None:
        """Connect to the database."""
        self._lock += 1
        if not self._is_open:
            self.conn = sqlite3.connect(self.file, detect_types=sqlite3.PARSE_DECLTYPES)
            self.cursor = self.conn.cursor()
            self._is_open = True

    @_on_rank_0
    def _close(self, *, force: bool = False, ensure_commit: bool = False) -> None:
        """Commits and closes connection if the _lock_count is at 0.

        Args:
            force (bool, optional): Force the closing of the connection. Defaults to False.
            force_commit (bool, optional): Force the commit of the connection. Defaults to False.
        """
        if force:
            self._lock = 0
        else:
            self._lock -= 1

        if self._lock <= 0 and self._is_open:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()
            self._is_open = False
            return

        if ensure_commit:
            self.conn.commit()

    @_on_rank_0
    @contextmanager
    def open(self, *, ensure_commit: bool = False) -> Generator[SQLTable]:
        """The open method is used as a context manager.

        Args:
            ensure_commit (bool, optional): Ensure that the connection is committed. Defaults to False.

        Example:
            >>> with db.table.open() as table:
            >>>     table.cursor.execute("SELECT * FROM database")
        """
        self._connect()
        yield self
        self._close(ensure_commit=ensure_commit)

    def commit_once(self, func) -> Callable:
        """Decorator to bundle changes to a single commit.

        Example:
            >>> @db.table.commit_once
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

    def _entry(self, entry_id: str) -> Entry:
        if entry_id not in self._entries:
            entry = Entry(self.path, entry_id)
            self._entries[entry_id] = entry
        return self._entries[entry_id]

    def _get_uids(self) -> list:
        """Get all simulation names in the database."""
        all_uids = list()
        for dir in os.listdir(self.path):
            if not os.path.isdir(os.path.join(self.path, dir)):
                continue
            if any(
                [i.endswith(".h5") for i in os.listdir(os.path.join(self.path, dir))]
            ):
                all_uids.append(dir)
        return all_uids

    # ----------------
    # METHODS
    # ----------------

    @with_connection
    @_on_rank_0
    def assert_table_exists(self) -> None:
        """Assert the SQL table exists for the database and update_times."""
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS update_times (id TEXT PRIMARY KEY,
                                                        update_time REAL)"""
        )
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS database (id TEXT PRIMARY KEY)"""
        )

    @with_connection
    @_on_rank_0
    def redo_table(self) -> None:
        """Rewrite all data into the database."""
        self.cursor.execute("""DROP TABLE IF EXISTS database""")
        self.cursor.execute("""DROP TABLE IF EXISTS update_times""")
        self.assert_table_exists()

        for id in self._get_uids():
            self.update_entry(
                id,
                self._entry(id).get_all_metadata(),
            )

    @_on_rank_0
    @with_connection
    def sync(self) -> None:
        """Sync the database with the file system."""
        all_ids = set(self._get_uids())

        self.cursor.execute("""SELECT id, update_time FROM update_times""")
        res = self.cursor.fetchall()

        for id, last_up_time in res:
            # remove entries that do not exist in the file system
            if id not in all_ids:
                self.cursor.execute("""DELETE FROM update_times WHERE id = ?""", (id,))
                self.cursor.execute("""DELETE FROM database WHERE id = ?""", (id,))
                continue

            # update entries that have been modified
            all_ids.remove(id)
            mtime = self._entry(id).modification_time
            if mtime > last_up_time:
                self.update_entry(id, self._entry(id).get_all_metadata())

        # add new entries
        for id in all_ids:
            self.update_entry(
                id,
                self._entry(id).get_all_metadata(),
            )

    @with_connection
    @_on_rank_0
    def update_entry(self, id: str, data: dict) -> None:
        """Update the entry in the database."""

        self.cursor.execute(
            """INSERT OR REPLACE INTO update_times (id, update_time) VALUES (?, ?)""",
            (id, time()),
        )

        # get columns of database table
        self.cursor.execute("PRAGMA table_info(database)")
        columns = self.cursor.fetchall()

        # check if column needs to be added
        for key, val in data.items():
            key = key.replace(".", self.DOT_REPLACEMENT)
            if not any(key == column[1] for column in columns):
                dtype = type(val)
                if isinstance(val, np.ndarray):
                    dtype = "ARRAY"
                elif isinstance(val, np.generic):
                    dtype = type(val.item())
                self.cursor.execute(
                    f"""ALTER TABLE database ADD COLUMN {key} {DATA_TYPES.get(dtype, "TEXT")}"""
                )

        # insert data into database table
        data.pop("id", None)
        sql = f"""
        INSERT INTO database (id, {", ".join(data.keys())}) 
        VALUES (:id, {", ".join(f":{key}" for key in data.keys())})
        ON CONFLICT(id) DO UPDATE SET {", ".join(f"{key} = excluded.{key}" for key in data.keys())}
        """
        data["id"] = id
        self.cursor.execute(sql, data)

    @with_connection
    @_on_rank_0
    def read_table(self, include_linked_sims: bool = False) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM database", self.conn)
        df.rename(columns=lambda x: x.replace(self.DOT_REPLACEMENT, "."), inplace=True)
        return df


class Entry:
    """Simulation entry in the database. Simplified version of Simulation
    class.
    """

    def __init__(self, path: str, id: str) -> None:
        self.path = path
        self.id = id
        self.h5file = os.path.join(path, id, f"{id}.h5")

    @property
    def parameters(self) -> dict:
        tmp_dict = dict()
        with open_h5file(self.h5file, "r") as file:
            try:
                tmp_dict.update(file["parameters"].attrs)
                for key in file["parameters"].keys():
                    tmp_dict.update({key: file[f"parameters/{key}"][()]})
            except KeyError:
                pass

        return tmp_dict

    @property
    def metadata(self) -> dict:
        tmp_dict = dict()
        with open_h5file(self.h5file, "r") as file:
            tmp_dict.update(file.attrs)

        return tmp_dict

    def get_all_metadata(self) -> dict:
        return {**self.parameters, **self.metadata}

    @property
    def modification_time(self) -> float:
        return os.path.getmtime(self.h5file)
