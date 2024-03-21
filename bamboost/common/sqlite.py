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
import logging
import sqlite3
from contextlib import contextmanager
from functools import wraps
from typing import Generator, Iterable

import numpy as np
from typing_extensions import Self

from bamboost.common.mpi import MPI

log = logging.getLogger(__name__)

DATA_TYPES = {
    "ARRAY": "ARRAY",
    "JSON": "JSON",
    np.datetime64: "DATETIME",
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "BOOL",
}


def parse_sqlite_type(val):
    if isinstance(val, np.ndarray):
        return "ARRAY"
    if isinstance(val, np.generic):
        return DATA_TYPES.get(type(val.item()), "TEXT")
    if isinstance(val, (dict, list, tuple, set)):
        return "JSON"
    return DATA_TYPES.get(type(val), "TEXT")


SYNC_TABLE = True


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, lambda arr: json.dumps(arr.tolist()))

adapt_numpy_number = lambda val: val.item()
sqlite3.register_adapter(np.int_, adapt_numpy_number)
sqlite3.register_adapter(np.float_, adapt_numpy_number)
sqlite3.register_adapter(np.datetime64, adapt_numpy_number)
sqlite3.register_adapter(bool, lambda val: int(val))
sqlite3.register_adapter(list, lambda val: json.dumps(val))

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
    def wrapper(self: SQLiteDatabase, *args, **kwargs):
        cursor: sqlite3.Cursor = self._cursor
        # check if cursor is available
        if not self._is_open or cursor is None:
            with self.open():
                return func(self, *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


def on_rank_0(func):
    """Decorator to ensure that the function is only executed on rank 0."""

    @wraps(func)
    def wrapper(self: SQLiteDatabase, *args, **kwargs):
        if self._comm.rank == 0:
            return func(self, *args, **kwargs)
        return None

    return wrapper


class SQLiteDatabase:
    """
    Class to handle sqlite databases.
    """

    def __init__(self, _comm=MPI.COMM_WORLD) -> None:
        if _comm.rank != 0:
            return
        self._comm = _comm
        self._conn = None
        self._cursor = None
        self._is_open = False
        self._lock_stack = 0

    @property
    def _lock_stack(self) -> int:
        return self.__lock_stack

    @_lock_stack.setter
    def _lock_stack(self, value: int) -> None:
        self.__lock_stack = value if value >= 0 else 0

    def connect(self) -> Self:
        self._lock_stack += 1
        if not self._is_open:
            log.debug(f"Connecting to {self.file}")
            self._conn = sqlite3.connect(
                self.file, detect_types=sqlite3.PARSE_DECLTYPES
            )
            self._cursor = self._conn.cursor()
            self._is_open = True
        return self

    def close(self, *, force: bool = False, ensure_commit: bool = False) -> Self:
        if force:
            self._lock_stack = 0
        else:
            self._lock_stack -= 1

        if self._lock_stack <= 0 and self._is_open:
            log.debug(f"Closing connection to {self.file}")
            self._conn.commit()
            self._conn.close()
            self._is_open = False
            return self

        if ensure_commit:
            self._conn.commit()

        return self

    def commit(self) -> None:
        self._conn.commit()
        return self

    @contextmanager
    def open(
        self,
        *,
        force_close: bool = False,
        ensure_commit: bool = False,
    ) -> Generator[Self]:
        """The open method is used as a context manager.

        Args:
            ensure_commit (bool, optional): Ensure that the connection is committed. Defaults to False.

        Example:
            >>> with index.open() as table:
            >>>     table._cursor.execute("SELECT * FROM database")
        """
        self.connect()
        yield self
        self.close(ensure_commit=ensure_commit, force=force_close)
