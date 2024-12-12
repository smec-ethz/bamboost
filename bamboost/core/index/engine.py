# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
"""
This module provides a class to handle sqlite databases.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable

import numpy as np

from bamboost import BAMBOOST_LOGGER

__all__ = [
    "get_sqlite_column_type",
    "SQLEngine",
]

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])

TYPE_MAP = {
    np.ndarray: "ARRAY",
    np.datetime64: "DATETIME",
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "BOOL",
}


def get_sqlite_column_type(val):
    dtype = type(val)
    if isinstance(val, np.generic):
        dtype = type(val.item())
    if dtype in TYPE_MAP:
        return TYPE_MAP[dtype]

    if isinstance(val, Iterable):
        return "JSON"
    if isinstance(val, dict):
        return "JSON"
    if isinstance(val, np.ndarray):
        return "ARRAY"


def _register_sqlite_adapters():
    # Converts np.array & Iterable to JSON when inserting
    sqlite3.register_adapter(np.ndarray, lambda arr: json.dumps(arr.tolist()))
    sqlite3.register_adapter(list, lambda val: json.dumps(val))
    sqlite3.register_adapter(dict, lambda val: json.dumps(val))
    sqlite3.register_adapter(tuple, lambda val: json.dumps(val))
    sqlite3.register_adapter(set, lambda val: json.dumps(val))

    # Numpy generic types
    def adapt_numpy_number(val):
        return val.item()

    sqlite3.register_adapter(np.int_, adapt_numpy_number)
    sqlite3.register_adapter(np.float64, adapt_numpy_number)
    sqlite3.register_adapter(np.datetime64, adapt_numpy_number)
    sqlite3.register_adapter(np.bool_, bool)


def _register_sqlite_converters(convert_arrays: bool = True):
    # Converts JSON to np.array & Iterable when selecting
    if convert_arrays:
        sqlite3.register_converter("ARRAY", lambda text: np.array(json.loads(text)))
    else:
        sqlite3.register_converter("ARRAY", lambda text: json.loads(text))
    sqlite3.register_converter("JSON", lambda text: json.loads(text))

    def convert_bool(val):
        # from now on, all bools should be stored as integers
        try:
            return bool(int(val))
        # for a while, booleans were stored as bytes
        # this exception handles the conversion of these bytes to bool for old databases
        except ValueError:
            return bool(int.from_bytes(val, "big"))

    sqlite3.register_converter("BOOL", convert_bool)


_register_sqlite_adapters()
_register_sqlite_converters()


class SQLEngine:
    """A simple wrapper for sqlite3 connections.

    The main benefit of this class is that it automatically commits changes
    and closes the connection when the context manager is exited.
    The connection can be chained without reopening the connection.

    Args:
        file: The path to the sqlite database file.
    """

    def __init__(self, file: str | Path):
        self.file = file
        self.conn: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None
        self._lock_stack: int = 0

    def connect(self) -> SQLEngine:
        self._lock_stack += 1
        if not self.conn:
            self.conn = sqlite3.connect(self.file, detect_types=sqlite3.PARSE_DECLTYPES)
            self.cursor = self.conn.cursor()
        return self

    def close(self) -> SQLEngine:
        self._lock_stack -= 1
        if self._lock_stack <= 0:
            self._lock_stack = 0
            try:
                self.conn.commit()
                self.conn.close()
            except AttributeError:
                pass
            finally:
                self.cursor = None
                self.conn = None
        return self

    def commit(self) -> None:
        if not self.conn:
            raise ValueError("Connection is not available. Please open the connection.")
        self.conn.commit()

    @contextmanager
    def open(self, *, force_commit: bool = False) -> Generator[SQLEngine, None, None]:
        self.connect()
        try:
            yield self
        finally:
            if force_commit:
                self.conn.commit()
            self.close()

    def execute(self, query: str, *args) -> sqlite3.Cursor:
        """Execute a query on the database.

        Args:
            query: The query to execute.
            args: The arguments to pass to the query.
        """
        if self.cursor is None:
            raise ValueError("Cursor is not available. Please open the connection.")

        self.cursor.execute(query, *args)
        return self.cursor
