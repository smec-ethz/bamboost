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
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from time import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import h5py
import pandas as pd

from bamboost import BAMBOOST_LOGGER, config
from bamboost.core.hdf5.file_handler import open_h5file
from bamboost.core.mpi import MPI

from .engine import Engine, create_all, schema

if TYPE_CHECKING:
    from mpi4py import MPI

log = BAMBOOST_LOGGER.getChild("database")


PREFIX = ".BAMBOOST-"
"prefix for databaseID identifier file"


class CacheAPI:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

        # Create the tables if they do not exist
        create_all(engine)

    def get_path(
        self,
        id: str,
        search_paths: Sequence[Union[str, Path]] = config.index.searchPaths,
    ) -> str:
        """Get the path of a database from its ID.

        Args:
            id (str): ID of the database

        Returns:
            str: path of the database
        """
        path = self.engine.execute(
            f"SELECT path FROM {self.__tablename__} WHERE id=?", (id,)
        ).fetchone()
        if path and _check_path(id, path[0]):
            return path[0]

        # if path is wrong, try to find it
        for root_dir in search_paths:
            log.debug(f"Searching for database {id} in {root_dir}")
            res = find(id, root_dir)
            if res:
                path = os.path.dirname(res[0])
                self.insert_path(id, path)
                return path

        raise FileNotFoundError(f"Database {id} not found on system.")

    def insert_path(self, id: str, path: str) -> None:
        """Insert a database path into the index.

        Args:
            id: ID of the database
            path: path of the database
        """
        path = os.path.abspath(path)
        self.engine.execute(
            f"INSERT OR REPLACE INTO {self.__tablename__} VALUES (?, ?)", (id, path)
        )

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
            f"SELECT id FROM {self.__tablename__} WHERE path=?", (path,)
        ).fetchone()
        if found_id is None:
            raise DatabaseNotFoundError(f"Database at {path} not found in index.")
        return found_id[0]

    def scan_paths_for_collections(
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

    def clean(self, purge: bool = False) -> CollectionsTable:
        """Clean the index from wrong paths.

        Args:
            purge: Also deletes the table of unmatching
                uid/path pairs. Defaults to False.
        """
        collections = self.engine.execute(
            f"SELECT id, path FROM {self.__tablename__}"
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

    def drop_path(self, id: str) -> None:
        """Drop a path from the index.

        Args:
            id: ID of the database
        """
        self.engine.execute(f"DELETE FROM {self.__tablename__} WHERE id=?", (id,))
        log.debug(f"Removed {id} from collections table")


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
