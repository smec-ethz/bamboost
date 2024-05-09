# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import logging
import os
import pkgutil
import sqlite3
import subprocess

import pandas as pd
import paramiko

from bamboost._config import config
from bamboost._sqlite_database import SQLiteHandler, with_connection
from bamboost.common.mpi import MPI
from bamboost.index import DatabaseTable, IndexAPI
from bamboost.manager import Manager
from bamboost.simulation import Simulation

log = logging.getLogger(__name__)

HOME = os.path.expanduser("~")
CACHE_DIR = os.path.join(HOME, ".cache", "bamboost")
# location off the index file on the remote server
REMOTE_INDEX = f"~/.local/share/bamboost/bamboost.db"


class Remote(IndexAPI, SQLiteHandler):
    """Access bamboost database of a remote server. The index is fetched using
    rsync over ssh. The `remote_name` can be a hostname or an IP address. Make
    sure that ssh keys are set and working, as there is no user authentication.
    The `skip_update` flag can be set to avoid fetching the index from the
    remote server.

    Args:
        - remote_name (str): The hostname or IP address of the remote server.
        - skip_update (bool): Flag to avoid fetching the index from the remote
          server. Default is False.

    Example:
        >>> remote = Remote("euler")
        >>> remote.read_table()
        returns a pandas DataFrame of the remote index.
        >>> remote["<id>"]
        returns a RemoteManager object for the given id.
    """

    def __new__(cls, *args, **kwargs):
        """Override the __new__ method to avoid the singleton pattern of IndexAPI."""
        return object.__new__(cls)

    def __init__(
        self, remote_name: str, skip_update: bool = False, *, home_path: str = None
    ) -> None:
        self.remote_name = remote_name
        self.local_path = os.path.join(CACHE_DIR, self.remote_name)
        os.makedirs(self.local_path, exist_ok=True)
        self.file = f"{self.local_path}/bamboost.db"

        if not skip_update:
            self._setup_ssh(home_path)

        # Initialize the SQLiteHandler
        SQLiteHandler.__init__(self, self.file)

    def _setup_ssh(self, home_path: str = None) -> None:
        """Set up the SSH and SFTP connection to the remote server."""
        # Setup the ssh connection using paramiko
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            # connect to remote server
            self.ssh.connect(self.remote_name)
            # open sftp session
            self.sftp = self.ssh.open_sftp()
            # get the home directory of the remote server and replace ~ with it
            if not home_path:
                home_path = self._get_remote_home()
            self.index_path = REMOTE_INDEX.replace("~", home_path)
            # fetch the index file
            self._fetch_index()
        except paramiko.ssh_exception.AuthenticationException:
            log.error(
                f"Could not connect to {self.remote_name}. Make sure ssh keys are set up."
            )
            self._cleanup()
            raise

    @property
    def ssh(self) -> paramiko.SSHClient:
        if hasattr(self, "_ssh"):
            return self._ssh
        raise AttributeError("SSH connection not set up.")

    @ssh.setter
    def ssh(self, ssh: paramiko.SSHClient) -> None:
        self._ssh = ssh

    @property
    def sftp(self) -> paramiko.SFTP:
        if hasattr(self, "_sftp"):
            return self._sftp
        raise AttributeError("SFTP connection not set up.")

    @sftp.setter
    def sftp(self, sftp: paramiko.SFTP) -> None:
        self._sftp = sftp

    def _cleanup(self) -> None:
        """Close the SFTP and SSH connections."""
        if hasattr(self, "sftp"):
            self.sftp.close()
        if hasattr(self, "ssh"):
            self.ssh.close()

    def __del__(self) -> None:
        self._cleanup()

    def _get_remote_home(self) -> str:
        """Get the home directory of the remote server."""
        stdin, stdout, stderr = self.ssh.exec_command("echo $HOME")
        return stdout.read().decode().strip()

    def _fetch_index(self) -> None:
        """Fetches the bamboost.db from the remote server."""
        # download the index file
        self.sftp.get(self.index_path, self.file)
        log.info(f"Index file fetched from {self.remote_name}")

    def _ipython_key_completions_(self) -> list[str]:
        ids = self.read_table()[["id", "path"]].values
        completion_keys = [
            f'{key} - {"..."+val[-25:] if len(val)>=25 else val}' for key, val in ids
        ]
        return completion_keys

    def __getitem__(self, id: str) -> RemoteManager:
        """Return a `RemoteManager` for the given id."""
        id = id.split(" - ")[0]
        return RemoteManager(id, remote=self)

    def get_manager(self, id: str, skip_update: bool = False) -> RemoteManager:
        return RemoteManager(id, remote=self, skip_update=skip_update)

    @with_connection
    def get_path(self, id: str) -> str:
        self._cursor.execute("SELECT path FROM dbindex WHERE id=?", (id,))
        return self._cursor.fetchone()[0]

    @with_connection
    def insert_local_path(self, id: str, path: str) -> None:
        try:
            self._cursor.execute(
                "UPDATE dbindex SET local_path=? WHERE id=?", (path, id)
            )
        except sqlite3.OperationalError:
            self._cursor.execute(
                "ALTER TABLE dbindex ADD COLUMN local_path TEXT DEFAULT NULL"
            )
            self._cursor.execute(
                "UPDATE dbindex SET local_path=? WHERE id=?", (path, id)
            )


class RemoteDatabaseTable(DatabaseTable):

    def sync(self) -> None:
        """Don't sync a remote database."""
        return None


class RemoteManager(Manager):
    """
    Manager class with remote functionality. Constructor takes an existing ID
    of a database on a remote server. The ssh connection must be set up to work
    without explicit user authentication. Data is lazily transferred using
    rsync.
    """

    # INFO: The initialization of this class takes around 8 seconds because the
    # home directory of the remote server needs to be fetched, read database
    # index, and get remote database csv file with db.df in it (see below).
    def __init__(
        self,
        id: str,
        remote: Remote,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """
        params
        skip_update: if True, does not lookup the new database on the remote:
        """
        self.UID = id
        self.remote = remote
        self.comm = comm
        self.path = os.path.join(self.remote.local_path, self.UID)
        log.info(f"Creating cache directory at {self.path}")
        os.makedirs(self.path, exist_ok=True)

        self.remote_path_db = self._index.get_path(self.UID)

        # check if path exists
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # Get the database ID file
        if not os.path.exists(f"{self.path}/.BAMBOOST-{self.UID}"):
            self.remote.sftp.get(
                f"{self.remote_path_db}/.BAMBOOST-{self.UID}",
                f"{self.path}/.BAMBOOST-{self.UID}",
            )

        self.UID = id
        self._index.insert_local_path(self.UID, self.path)

        # Update the SQL table for the database
        with self._index.open():
            self._table.create_database_table()
            # self._table.sync()

    def _repr_html_(self) -> str:
        html_string = pkgutil.get_data(__name__, "../html/manager.html").decode()
        icon = pkgutil.get_data(__name__, "../html/icon.txt").decode()
        return (
            html_string.replace("$ICON", icon)
            .replace("$db_path", self.path)
            .replace("$db_uid", "euler-" + self.UID)
            .replace("$db_size", str(len(self)))
        )

    def _get_uids(self) -> list:
        """Override the simulation list to fetch from databasetable."""
        return self._index.fetch(f"SELECT id FROM db_{self.UID}")

    @property
    def _index(self) -> IndexAPI:
        return self.remote

    @property
    def _table(self) -> RemoteDatabaseTable:
        return RemoteDatabaseTable(self.UID, _index=self._index)

    def get_view(self, include_linked_sims: bool = False) -> pd.DataFrame:
        df = super().get_view(include_linked_sims)
        df.insert(1, "cached", False)
        for id in os.listdir(self.path):
            if id in df["id"].values:
                df.loc[df["id"] == id, "cached"] = True

        opts = config.get("options", {})
        if "sort_table_key" in opts:
            df.sort_values(
                opts.get("sort_table_key", "id"),
                ascending=opts.get("sort_table_order", "asc") == "asc",
                inplace=True,
            )
        return df

    def sim(self, uid, return_writer: bool = False):
        """Return simulation object.

        Args:
            - uid (str): the unique id of the sim to be transferred.
            - return_writer (bool): Flag to indicate whether to return a writer
              object. Default is False.

        This method checks if the data for the given uid is already in the
        local cache. If not, it transfers the data from a remote location using
        rsync. The method then calls the superclass method to perform further
        operations on the transferred data.
        """
        # Check if data is already in cache
        if os.path.exists(f"{self.path}/{uid}"):
            log.info(f"Data for {uid} already in cache")
            return RemoteSimulation(uid, self)

        # Transfer data using rsync
        log.info(f"Data not in cache. Transferring data for {uid} from {self.remote}")
        self.rsync(uid)

        return RemoteSimulation(uid, self)

    def rsync(self, uid: str) -> None:
        """Transfer data using rsync."""
        subprocess.call(
            [
                "rsync",
                "-r",
                f"{self.remote.remote_name}:{self.remote_path_db}/{uid}",
                f"{self.path}",
            ],
            stdout=subprocess.PIPE,
        )
        log.info(f"Data for {uid} synced with {self.path}")


class RemoteSimulation(Simulation):
    def __init__(self, uid: str, manager: RemoteManager) -> None:
        super().__init__(uid, manager.path, _db_id=manager.UID)
        self.manager = manager

    def sync(self) -> RemoteSimulation:
        """Sync the simulation data with the remote server."""
        self.manager.rsync(self.uid)
        return self

    def get_full_uid(self) -> str:
        return f"{self.manager.UID}:{self.uid}"


if __name__ == "__main__":
    pass
