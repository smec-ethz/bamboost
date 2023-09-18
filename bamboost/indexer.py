# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import os
import subprocess
import logging
import json

log = logging.getLogger(__name__)


class Indexer:
    """Handling the collection of databases of the system.
    If a Database is accessed with its UID, the indexer looks in the following
    places:
        1. `~/.config/bamboost/database_index.json`
        2. search the current working directory
        3. search in specified directories (`~/.config/bamboost/directories.toml`)
        4. search in home directory
    """
    _prefix = '.BAMBOOST-'

    def __init__(self) -> None:
        self.home = os.path.expanduser('~')
        self.dir_config = os.path.join(self.home, '.config', 'bamboost')
        self.database_index = os.path.join(self.dir_config, 'database_index.json')
        self.known_paths = os.path.join(self.dir_config, 'known_paths.json')
        os.makedirs(self.dir_config, exist_ok=True)
        if not os.path.isfile(self.database_index):
            with open(self.database_index, 'w') as file:
                file.write(json.dumps({}, indent=4))
        if not os.path.isfile(self.known_paths):
            with open(self.known_paths, 'w') as file:
                file.write(json.dumps([], indent=4))

    def _get_index_dict(self) -> dict:
        with open(self.database_index, 'r') as file:
            try:
                return json.loads(file.read())
            except json.JSONDecodeError:
                return {}

    def _write_index_dict(self, index: dict) -> None:
        with open(self.database_index, 'w') as file:
            file.write(json.dumps(index, indent=4))

    def _get_known_paths(self) -> list:
        with open(self.known_paths, 'r') as file:
            return json.loads(file.read())

    def record_database(self, uid: str, path: str) -> None:
        """Record a database in `database_index.json`

        Args:
            uid: the uid of the database
            path: the path of the database
        """
        index = self._get_index_dict()
        index[uid] = path
        self._write_index_dict(index)

    def get_path(self, uid: str) -> str:
        """Find the path of a database specified by its UID.

        Args:
            uid: the UID of the database
        """
        # check in index
        index = self._get_index_dict()
        if uid in index.keys():
            path = index[uid]
            if self._check_path(uid, path):
                return path
            else:
                del index[uid]
                self._write_index_dict(index)

        # check known paths
        known_paths = self._get_known_paths()
        for path in known_paths:
            res = self.find(uid, root_dir=path)
            if res:
                return res

    def find(self, uid, root_dir) -> list:
        """Find the database with UID under given root_dir.

        Args:
            uid: UID to search for
            root_dir: root directory for search
        """
        if os.name=='posix':
            paths = self._find_posix(uid, root_dir)
        else:
            paths = self._find_python(uid, root_dir)
        if len(paths) > 0:
            log.warning(f'Multiple paths found for UID {uid}:\n{paths}')
        return paths[0]  # return the first found if multiple are found

    def _find_posix(self, uid, root_dir) -> list:
        stdout = subprocess.run(['find', root_dir, '-iname', self.uid2(uid),
                                 '-not', '-path', '*/\.git/*'], capture_output=True)
        paths_found = stdout.decode('utf-8').splitlines()
        return paths_found

    def _find_python(self, uid, root_dir) -> list:
        pass

    def uid2(self, uid) -> str:
        return f'{self._prefix}{uid}'
        
    def _check_path(self, uid: str, path: str) -> bool:
        """Check if path is going to the correct database"""
        if f'{self._prefix}{uid}' in os.listdir(path):
            return True
        else:
            return False

