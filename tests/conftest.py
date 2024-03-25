import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from bamboost import index
from bamboost.manager import Manager


def make_persistent_temp_manager():
    temp_dir = tempfile.mkdtemp()
    db = Manager(path=temp_dir)
    booleans = {
        "1": True,
        "2": False,
        "3": True,
    }

    @index.Index.commit_once
    def create_sims(booleans: list):
        for args in zip([1, 2, 3], booleans, ["a", "b", "c"]):
            db.create_simulation(
                uid=args[1],
                parameters={
                    "int": args[0],
                    "float": 1.0,
                    "str": args[2],
                    "boolean": booleans[args[1]],
                    "boolean2": False,
                    "array": np.array([1, 2, 3]),
                },
            )

    create_sims(booleans)

    return db.path


temp_manager_persistent: str = make_persistent_temp_manager()


def pytest_sessionstart(session):
    """Setup tmp config directory."""
    tempdir = tempfile.mkdtemp()
    index.CONFIG_DIR = tempdir
    index.DATABASE_INDEX = os.path.join(index.CONFIG_DIR, "database_index.json")
    index.KNOWN_PATHS = os.path.join(index.CONFIG_DIR, "known_paths.json")
    index.LOCAL_DIR = os.path.join(index.CONFIG_DIR, "local")

    # Create config files if they don't exist
    os.makedirs(index.CONFIG_DIR, exist_ok=True)
    os.makedirs(index.LOCAL_DIR, exist_ok=True)
    if not os.path.isfile(index.DATABASE_INDEX):
        with open(index.DATABASE_INDEX, "w") as file:
            file.write(json.dumps({}, indent=4))
    if not os.path.isfile(index.KNOWN_PATHS):
        with open(index.KNOWN_PATHS, "w") as file:
            file.write(json.dumps([], indent=4))

    # change path of sqlite database
    index.Index.__init__()
    print(index.Index.file)


def pytest_sessionfinish(session, exitstatus):
    """Remove tmp config directory again."""
    shutil.rmtree(index.CONFIG_DIR)
    shutil.rmtree(temp_manager_persistent)
