# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import ast
import io
import json
import os
import sqlite3
from datetime import datetime
from functools import wraps
from typing import Tuple

import numpy as np
import pandas as pd

from bamboost import index
from bamboost.common.file_handler import open_h5file
from bamboost.common.mpi import MPI

from ..manager import Manager

DATA_TYPES = {
    "ARRAY": "ARRAY",
    "JSON": "JSON",
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "BOOL",
}


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, lambda arr: json.dumps(arr.tolist()))

adapt_numpy_number = lambda val: val.item()
sqlite3.register_adapter(np.int_, adapt_numpy_number)
sqlite3.register_adapter(np.float_, adapt_numpy_number)
sqlite3.register_adapter(np.datetime64, adapt_numpy_number)
sqlite3.register_adapter(bool, lambda val: int(val))

# Converts TEXT to np.array when selecting
sqlite3.register_converter("ARRAY", lambda text: np.array(json.loads(text)))
sqlite3.register_converter("JSON", lambda text: json.loads(text))
sqlite3.register_converter("BOOL", lambda val: bool(val))


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
            with self.open:
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
    """SQLite file for a bamoost database."""

    DOT_REPLACEMENT = "DOT"
    _instances = {}

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
        self.id = id
        self.path = path if path is not None else index.get_path(id)
        self._comm = _comm
        # only rank 0 will interact with the table
        if self._comm.rank != 0:
            return

        self.file = os.path.abspath(
            os.path.join(self.path, f".{self.id}.sqlite")
        )
        self.conn = None
        self.cursor = None
        self._is_open = False
        self._lock: int = 0

        self._entries = dict()

    @_on_rank_0
    def _connect(self) -> None:
        """Connect to the database."""
        self.conn = sqlite3.connect(self.file, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()
        self._is_open = True

    @_on_rank_0
    def _close(self) -> None:
        """Close the database."""
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        self._is_open = False

    @property
    @_on_rank_0
    def open(self) -> SQLTable:
        """For readability, the open method is used as a context manager."""
        return self

    @_on_rank_0
    def __enter__(self) -> sqlite3.Cursor:
        self._lock += 1
        if not self._is_open:
            self._connect()
        return self.cursor

    @_on_rank_0
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock -= 1
        if self._lock <= 0:
            self._close()

    def _entry(self, entry_id: str) -> Entry:
        def new_entry(entry_id):
            entry = Entry(self.path, entry_id)
            self._entries[entry_id] = entry
            return entry

        return self._entries.get(entry_id, new_entry(entry_id))

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
                self._entry(id).modification_time,
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
            if mtime != last_up_time:
                self.update_entry(id, self._entry(id).get_all_metadata(), mtime)

        # add new entries
        for id in all_ids:
            self.update_entry(
                id,
                self._entry(id).get_all_metadata(),
                self._entry(id).modification_time,
            )

    @with_connection
    @_on_rank_0
    def update_entry(self, id: str, data: dict, modification_time: float) -> None:
        """Update the entry in the database."""

        self.cursor.execute(
            """INSERT OR REPLACE INTO update_times (id, update_time) VALUES (?, ?)""",
            (id, modification_time),
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
        self.cursor.execute(
            f"""INSERT OR REPLACE INTO database ({", ".join([i.replace('.', self.DOT_REPLACEMENT) for i in data.keys()])}) VALUES ({", ".join(["?"]*len(data))})""",
            tuple(data.values()),
        )

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
