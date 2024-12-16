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

import os
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from time import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import h5py
import pandas as pd
from sqlalchemy import Connection, Row, create_engine, select, update
from sqlalchemy.dialects.sqlite import insert

from bamboost import BAMBOOST_LOGGER, config
from bamboost.core.hdf5.file_handler import open_h5file
from bamboost.core.mpi import MPI

from .schema import collections, create_all, parameters, simulations

if TYPE_CHECKING:
    from mpi4py import MPI

log = BAMBOOST_LOGGER.getChild("database")


PREFIX = ".BAMBOOST-"
"prefix for databaseID identifier file"


_APIMethod = TypeVar("_APIMethod", bound=Callable[..., Any])


def _with_mpi(make_changes: bool = False) -> Callable[[_APIMethod], _APIMethod]:
    def decorator(method: _APIMethod) -> _APIMethod:
        """Decorator to handle MPI communication in methods.

        Args:
            method: method to decorate
        """

        @wraps(method)
        def inner(instance: CacheAPI, *args, **kwargs):
            comm = MPI.COMM_WORLD

            # handle MPI
            if comm.rank == 0:
                with instance.transaction(make_changes=make_changes):
                    result = method(instance, *args, **kwargs)

            return comm.bcast(result, root=0)

        return cast(_APIMethod, inner)

    return decorator


class CacheAPI:
    def __init__(self, file: str | Path) -> None:
        self.file = file
        self.engine = create_engine(f"sqlite:///{file}")
        self.conn: Connection | None = None
        self._context_stack: int = 0
        self._needs_commit: bool = False

        # Create the tables if they do not exist
        create_all(self.engine)

    def connect(self, *, make_changes: bool = False) -> Connection:
        self._context_stack += 1
        if getattr(self.conn, "closed", True):  # if self.conn.closed is True
            self.conn = self.engine.connect()
            log.debug(f"Opened connection to {self.file}")

        return cast(Connection, self.conn)

    def close(self) -> CacheAPI:
        self._context_stack -= 1

        if self._context_stack <= 0:
            self._context_stack = 0

            try:
                if self._needs_commit:
                    self.conn.commit()
            except AttributeError:
                pass
            finally:
                self.conn.close()
                log.debug(f"Closed connection to {self.file}")

        return self

    @contextmanager
    def transaction(
        self, *, make_changes: bool = False, force_commit: bool = False
    ) -> Generator[Connection, None, None]:
        conn = self.connect()
        self._needs_commit = self._needs_commit or make_changes

        try:
            yield conn
        finally:
            if force_commit:
                conn.commit()
            self.close()

    @_with_mpi()
    def read_all(self) -> Sequence[Row[Any]]:
        query = select(collections)
        return self.conn.execute(query).fetchall()

    @_with_mpi()
    def simulations(self, collection_id: str) -> Sequence[Row[Any]]:
        query = select(simulations).where(simulations.c.collection_id == collection_id)
        return self.conn.execute(query).fetchall()

    @_with_mpi()
    def get_path(
        self,
        uid: str,
    ) -> str:
        """Get the path of a collection from its UID.

        Args:
            id: UID of the collection

        Returns:
            str: path of the collection
        """
        query = select(collections.c.path).where(collections.c.id == uid)
        return self.conn.execute(query).scalar_one()

    @_with_mpi(make_changes=True)
    def insert_path(self, uid: str, path: str) -> None:
        """Insert a collection path into the cache.

        Args:
            id: ID of the collection
            path: path of the collection
        """
        stmt = (
            insert(collections)
            .values(id=uid, path=path)
            .on_conflict_do_update(index_elements=["id"], set_={"path": path})
        )
        self.conn.execute(stmt)

    @_with_mpi()
    def get_uid(self, path: str) -> str:
        """Get the UID of a collection from its path.

        Args:
            path: path to the collection

        Returns:
            str: UID of the collection

        Raises:
            DatabaseNotFoundError: if the collection is not found in the index
        """
        stmt = select(collections.c.id).where(collections.c.path == path)
        return self.conn.execute(stmt).scalar_one()

    @_with_mpi(make_changes=True)
    def drop_collection(self, uid: str) -> None:
        """Drop a collection from the cache.

        Args:
            uid: UID of the collection
        """
        stmt = collections.delete().where(collections.c.id == uid)
        self.conn.execute(stmt)

    @_with_mpi()
    def get_collection(self, uid: str) -> tuple[Sequence, Sequence[Row[Any]]]:
        stmt = (
            select(
                simulations.c.id.label("simulation_id"),
                simulations.c.name.label("simulation_name"),
                simulations.c.created_at,
                simulations.c.modified_at,
                parameters.c.key.label("parameter_key"),
                parameters.c.value.label("parameter_value"),
            )
            .select_from(
                simulations.join(
                    collections, simulations.c.collection_id == collections.c.id
                ).outerjoin(parameters, simulations.c.id == parameters.c.simulation_id)
            )
            .where(collections.c.id == uid)
        )
        cursor_result = self.conn.execute(stmt)
        return cursor_result.cursor.description, cursor_result.all()


class CollectionEntry:
    def __init__(self, id: str, collections_table: CollectionsTable) -> None:
        self.id = id
        self.table = collections_table
        self.engine = collections_table.engine

    def get_df(self) -> pd.DataFrame:
        query: str = """
            SELECT 
                s.id AS simulation_id,
                s.name AS simulation_name,
                s.created_at,
                s.updated_at,
                s.status,
                p.key AS parameter_key,
                p.value AS parameter_value
            FROM
                Simulations s
            INNER JOIN
                Collections c ON s.collection_id = c.id
            LEFT JOIN
                Parameters p ON s.id = p.simulation_id
            WHERE
                c.id = ?;
        """
        result = pd.read_sql_query(query, self.engine.conn, params=[self.id])
        return result


class CollectionTable:
    """Class to manage the table of a database. Multiton pattern. One table per
    database.
    """

    def __init__(
        self,
        id: str,
        database: CollectionsTable,
        engine: sql.Engine | None = None,
    ):
        self.id: str = id
        self.database: CollectionsTable = database
        self.engine: sql.Engine = engine or database.engine
        self.path: str = self.database.get_path(self.id)

        self.TABLENAME = f"coll_{self.id}"
        self.TABLENAME_UPDATE_TIME = f"coll_{self.id}_t"

        self._entries = {}
        self._initialized = True
        self.create_database_table()

    def read_table(self) -> pd.DataFrame:
        """Read the table of the database.

        Returns:
            Table of the database
        """
        df = pd.read_sql_query(f"SELECT * FROM {self.TABLENAME}", self.engine.conn)
        # drop "hidden" columns which start with _
        # TODO: is this necessary?
        df = df.loc[:, ~df.columns.str.startswith("_")]
        return df

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
        values = cursor.fetchone()
        description = cursor.description
        series = pd.Series(
            values, index=[description[0] for description in description]
        )
        return series

    def read_columns(self, *columns: str) -> pd.DataFrame:
        """Read columns from the database.

        Args:
            *columns: columns to read

        Returns:
            pd.DataFrame: columns from the database
        """
        cursor = self.engine.execute(
            f"SELECT {', '.join(f'[{i}]' for i in columns)} FROM {self.TABLENAME}"
        )
        df = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
        return df

    def drop_table(self) -> None:
        """Drop the table of the database."""
        self.engine.execute(f"DROP TABLE {self.TABLENAME}")
        self.engine.execute(f"DROP TABLE {self.TABLENAME_UPDATE_TIME}")

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

    def update_entry(self, entry_id: str, data: dict) -> None:
        """Update an entry in the database.

        Args:
            entry_id (str): ID of the entry
            data (dict): data to update
        """
        cursor = self.engine.cursor
        # return if data is empty
        if not data:
            return

        # get columns of table
        cursor.execute(f"PRAGMA table_info({self.TABLENAME})")
        cols = cursor.fetchall()

        # check if columns exist
        for key, val in data.items():
            if any(key == column[1] for column in cols):
                continue
            dtype = sql.get_sqlite_column_type(val)
            cursor.execute(f"ALTER TABLE {self.TABLENAME} ADD COLUMN [{key}] {dtype}")

        # insert data into table
        keys = ", ".join([f"[{key}]" for key in data.keys()])
        placeholders = ", ".join(["?" for _ in data.keys()])
        updates = ", ".join([f"[{key}] = excluded.[{key}]" for key in data.keys()])

        query = f"""
            INSERT INTO {self.TABLENAME} (id, {keys})
            VALUES (?, {placeholders})
            ON CONFLICT(id) DO UPDATE SET {updates}
        """
        log.debug("execute query: {}".format(query))
        cursor.execute(query, (entry_id, *data.values()))

        # update update time
        cursor.execute(
            f"INSERT OR REPLACE INTO {self.TABLENAME_UPDATE_TIME} VALUES (?, ?)",
            (entry_id, time()),
        )

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


def get_known_paths() -> set[Path]:
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
