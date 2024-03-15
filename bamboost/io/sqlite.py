# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations
from datetime import datetime

import io
import os
import sqlite3
from typing import Tuple

import numpy as np

DATA_TYPES = {
    "array": "array",
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
}


# Register adapters for numpy types
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def adapt_numpy_number(val):
    return val.item()


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_adapter(np.int_, adapt_numpy_number)
sqlite3.register_adapter(np.float_, adapt_numpy_number)
sqlite3.register_adapter(np.datetime64, adapt_numpy_number)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


class SQLiteInterface:
    """SQLite file for a bamoost database."""

    def __init__(self, path: str, id: str) -> None:
        self.id = id
        self.path = path
        self.file = os.path.abspath(os.path.join(self.path, f".{self.id}.sqlite"))

    def connect(self) -> None:
        """Connect to the database."""
        self.conn = sqlite3.connect(
            self.file, detect_types=sqlite3.PARSE_DECLTYPES
        )
        self.cursor = self.conn.cursor()

    def close(self) -> None:
        """Close the database."""
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def __enter__(self) -> sqlite3.Cursor:
        self.connect()
        return self.cursor

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def update_sql_table(self) -> None:
        """Update the SQL table for the database."""
        ...

    def assert_table_exists(self) -> None:
        """Assert the SQL table exists for the database and update_times."""
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS update_times (id TEXT PRIMARY KEY,
                                                        update_time REAL)"""
        )
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS database (id TEXT PRIMARY KEY)"""
        )

    def is_up_to_date(self, id: str) -> Tuple[bool, float]:
        self.cursor.execute(
            """SELECT update_time FROM update_times WHERE id = ?""", (id,)
        )
        res = self.cursor.fetchone()

        # read modification time of h5 file
        h5file = os.path.join(self.path, id, f"{id}.h5")
        modification_time = os.path.getmtime(h5file)
        if res is None or res[0] != modification_time:
            return False, modification_time

        return True, modification_time

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
            key = key.replace(".", "$")
            if not any(key == column[1] for column in columns):
                dtype = type(val)
                if isinstance(val, np.ndarray):
                    dtype = "array"
                elif isinstance(val, np.generic):
                    dtype = type(val.item())
                self.cursor.execute(
                    f"""ALTER TABLE database ADD COLUMN {key} {DATA_TYPES.get(dtype, "TEXT")}"""
                )

        # insert data into database table
        self.cursor.execute(
            f"""INSERT OR REPLACE INTO database ({", ".join([i.replace('.', '$') for i in data.keys()])}) VALUES ({", ".join(["?"]*len(data))})""",
            tuple(data.values()),
        )
