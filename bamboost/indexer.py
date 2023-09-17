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
import json


class Indexer:
    """Handling the collection of databases of the system.
    If a Database is accessed with its UID, the indexer looks in the following
    places:
        1. `~/.config/bamboost/database_index.json`
        2. search the current working directory
        3. search in specified directories (`~/.config/bamboost/directories.toml`)
        4. search in home directory
    """
    def __init__(self) -> None:
        self.home = os.path.expanduser('~')
        self.dir_config = os.path.join(self.home, '.config', 'bamboost')
        self.database_index = os.path.join(self.dir_config, 'database_index.json')
        os.makedirs(self.dir_config, exist_ok=True)
        if not os.path.isfile(self.database_index):
            with open(self.database_index, 'w') as file:
                file.write(json.dumps({}, indent=4))

    def _get_index_dict(self) -> dict:
        with open(self.database_index, 'r') as file:
            try:
                return json.loads(file.read())
            except json.JSONDecodeError:
                return {}

    def _write_index_dict(self, index: dict) -> None:
        with open(self.database_index, 'w') as file:
            file.write(json.dumps(index, indent=4))

    def record_database(self, uid: str, path: str) -> None:
        """Record a database in `database_index.json`

        Args:
            uid: the uid of the database
            path: the path of the database
        """
        index = self._get_index_dict()
        index[uid] = path
        self._write_index_dict(index)

        

