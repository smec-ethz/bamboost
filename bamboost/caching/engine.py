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
from abc import ABC
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Sequence, TypeVar, cast

import numpy as np

from bamboost import BAMBOOST_LOGGER
from bamboost.core.mpi import MPI

__all__ = [
    "get_sqlite_column_type",
    "Engine",
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


# Converts JSON to np.array & Iterable when selecting
sqlite3.register_converter("ARRAY", lambda text: np.array(json.loads(text)))
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


class Engine:
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
        self._context_level: int = 0

    def connect(self) -> Engine:
        self._context_level += 1
        if not self.conn:
            self.conn = sqlite3.connect(self.file, detect_types=sqlite3.PARSE_DECLTYPES)
            self.cursor = self.conn.cursor()
        return self

    def close(self) -> Engine:
        self._context_level -= 1
        if self._context_level <= 0:
            self._context_level = 0

            try:
                self.conn.commit()
            except AttributeError:
                pass
            finally:
                self.conn.close()
                self.cursor = None
                self.conn = None

        return self

    def commit(self) -> None:
        if not self.conn:
            raise ValueError("Connection is not available. Please open the connection.")
        self.conn.commit()

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

    @contextmanager
    def transaction(
        self, *, force_commit: bool = False
    ) -> Generator[Engine, None, None]:
        self.connect()
        try:
            yield self
        finally:
            if force_commit:
                self.conn.commit()
            self.close()


# Define a TypeVar for a bound method of IndexAPI
IndexAPIMethod = TypeVar("IndexAPIMethod", bound=Callable[..., Any])


def _get(method: IndexAPIMethod) -> IndexAPIMethod:
    """Decorator for methods in IndexAPI to handle MPI broadcasting.

    Args:
        method: A method of IndexAPI, where the first argument is `self`.

    Returns:
        A decorated function that takes the same arguments as `method`.
    """
    comm = MPI.COMM_WORLD

    @wraps(method)
    def inner(self: _Table, *args, **kwargs):
        # handle MPI
        if comm.rank == 0:
            with self.engine.transaction():
                result = method(self, *args, **kwargs)

        return comm.bcast(result, root=0)

    return cast(IndexAPIMethod, inner)


def _write(method: IndexAPIMethod) -> IndexAPIMethod:
    comm = MPI.COMM_WORLD

    @wraps(method)
    def inner_root(self: _Table, *args, **kwargs):
        with self.engine.transaction():
            method(self, *args, **kwargs)

    def inner_off_root(*_args, **_kwargs):
        pass

    if comm.rank == 0:
        return cast(IndexAPIMethod, inner_root)
    else:
        return cast(IndexAPIMethod, inner_off_root)


class _Table(ABC):
    """A table in the relational database."""

    __tablename__: str
    __schema__: Sequence[str]

    def __init__(self, engine: Engine):
        self.engine = engine
        self.column_names = tuple(i.split()[0] for i in self.__schema__)

    def __repr__(self) -> str:
        return f"Table {self.__tablename__} in {self.engine.file}"

    @_write
    def create_if_not_exists(self) -> None:
        """Create the index table if it does not exist."""
        self.engine.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.__tablename__} (
                {", ".join(self.__schema__)}
            )"""
        )

    @_get
    def rows(self) -> list[tuple]:
        """Get all rows from the table.

        Returns:
            list[tuple]: all rows from the table
        """
        return self.engine.execute(f"SELECT * FROM {self.__tablename__}").fetchall()

    @_get
    def columns(self, *columns: str) -> list[tuple]:
        """Get columns from the table.

        Args:
            *columns: columns to get

        Returns:
            list[tuple]: columns from the table
        """
        if not columns:
            columns = self.column_names

        return self.engine.execute(
            f"SELECT {', '.join(columns)} FROM {self.__tablename__}"
        ).fetchall()


class _Collections(_Table):
    """Main class to manage the database of bamboost collections."""

    __tablename__ = "Collections"
    __schema__ = (
        "id TEXT PRIMARY KEY",
        "path TEXT",
    )


class _Simulations(_Table):
    """Table to cache all known simulations."""

    __tablename__ = "Simulations"
    __schema__ = (
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "collection_id TEXT NOT NULL",
        "name TEXT NOT NULL",
        "description TEXT",
        "created_at DATETIME",
        "updated_at DATETIME",
        "status TEXT",
        "FOREIGN KEY (collection_id) REFERENCES Collections (id) ON DELETE CASCADE",
    )


class _Parameters(_Table):
    """Table to cache all known parameters of simulations."""

    __tablename__ = "Parameters"
    __schema__ = (
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "simulation_id INTEGER NOT NULL",
        "key TEXT NOT NULL",
        "value JSON NOT NULL",
        "FOREIGN KEY (simulation_id) REFERENCES Simulations (id) ON DELETE CASCADE",
    )


schema = (_Collections, _Simulations, _Parameters)


def create_all(engine: Engine) -> None:
    """Create all tables in the database.

    Args:
        engine: The engine to use for the database.
    """
    for table in schema:
        table(engine).create_if_not_exists()
