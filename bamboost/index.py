# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
"""Module to manage the database index and its ID's."""

from __future__ import annotations

import json
import logging
import os
import subprocess

log = logging.getLogger(__name__)


# Define directories
# ------------------

HOME = os.path.expanduser("~")
CONFIG_DIR = os.path.join(HOME, ".config", "bamboost")
DATABASE_INDEX = os.path.join(CONFIG_DIR, "database_index.json")
KNOWN_PATHS = os.path.join(CONFIG_DIR, "known_paths.json")
PREFIX = ".BAMBOOST-"

# Create config files if they don't exist
os.makedirs(CONFIG_DIR, exist_ok=True)
if not os.path.isfile(DATABASE_INDEX):
    with open(DATABASE_INDEX, "w") as file:
        file.write(json.dumps({}, indent=4))
if not os.path.isfile(KNOWN_PATHS):
    with open(KNOWN_PATHS, "w") as file:
        file.write(json.dumps([], indent=4))


# Module level functions
# ----------------------


def get_index_dict() -> dict:
    """Returns a dictionary of all known databases."""
    with open(DATABASE_INDEX, "r") as file:
        try:
            return json.loads(file.read())
        except json.JSONDecodeError:
            return {}


def _write_index_dict(index: dict) -> None:
    """Write the database index."""
    with open(DATABASE_INDEX, "w+") as file:
        file.write(json.dumps(index, indent=4))


def get_known_paths() -> list:
    with open(KNOWN_PATHS, "r") as file:
        return json.loads(file.read())


def _find_posix(uid, root_dir) -> list:
    """Find function using system `find` on linux."""
    completed_process = subprocess.run(
        ["find", root_dir, "-iname", uid2(uid), "-not", "-path", "*/\.git/*"],
        capture_output=True,
    )
    paths_found = completed_process.stdout.decode("utf-8").splitlines()
    return paths_found


def _find_python(uid, root_dir) -> list:
    """Some find function for Windows or other if `find` is not working.

    TODO: to be implemented
    """
    pass


def uid2(uid) -> str:
    return f"{PREFIX}{uid}"


def get_uid_from_path(path: str) -> str:
    """Returns the UID found in the specified path."""
    for file in os.listdir(path):
        if file.startswith(".BAMBOOST"):
            return file.split("-")[1]
    raise FileNotFoundError("No UID file found at specified path.")


def _check_path(uid: str, path: str) -> bool:
    """Check if path is going to the correct database"""
    if not os.path.exists(path):
        return False
    if f"{PREFIX}{uid}" in os.listdir(path):
        return True
    return False


def record_database(uid: str, path: str) -> None:
    """Record a database in `database_index.json`

    Args:
        uid: the uid of the database
        path: the path of the database
    """
    index = get_index_dict()
    index[uid] = path
    _write_index_dict(index)


def get_path(uid: str) -> str:
    """Find the path of a database specified by its UID.

    Args:
        uid: the UID of the database
    """
    # check in index
    index = get_index_dict()
    if uid in index.keys():
        path = index[uid]
        if _check_path(uid, path):
            return path
        else:
            del index[uid]
            _write_index_dict(index)

    # check known paths
    known_paths = get_known_paths()
    for path in known_paths:
        res = find(uid, root_dir=path)
        if res:
            path = os.path.dirname(res[0])
            record_database(uid, path)
            return path

    # check home
    res = find(uid, HOME)
    if res:
        path = os.path.dirname(res[0])
        record_database(uid, path)
        return path

    raise FileNotFoundError(f"Database {uid} not found on system.")


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


def clean() -> None:
    """Clean the database index from wrong paths."""
    index = get_index_dict()
    clean_index = {uid: path for uid, path in index.items() if _check_path(uid, path)}
    _write_index_dict(clean_index)


def create_index() -> None:
    """Create database index from known paths."""
    known_paths = get_known_paths()
    index = get_index_dict()
    for path in known_paths:
        completed_process = subprocess.run(
            ["find", path, "-iname", f"{PREFIX}*", "-not", "-path", "*/\.git/*"],
            capture_output=True,
        )
        databases_found = completed_process.stdout.decode("utf-8").splitlines()
        for database in databases_found:
            name = os.path.basename(database)
            uid = name.split("-")[1]
            index[uid] = os.path.dirname(database)
    _write_index_dict(index)
