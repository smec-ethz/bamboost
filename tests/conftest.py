import pytest
import tempfile
import os
import json
import shutil
from bamboost import index


def pytest_sessionstart(session):
    """Setup tmp config directory."""
    tempdir = tempfile.mkdtemp()
    index.CONFIG_DIR = tempdir
    index.DATABASE_INDEX = os.path.join(index.CONFIG_DIR, 'database_index.json')
    index.KNOWN_PATHS = os.path.join(index.CONFIG_DIR, 'known_paths.json')
    print(tempdir)

    # Create config files if they don't exist
    os.makedirs(index.CONFIG_DIR, exist_ok=True)
    if not os.path.isfile(index.DATABASE_INDEX):
        with open(index.DATABASE_INDEX, 'w') as file:
            file.write(json.dumps({}, indent=4))
    if not os.path.isfile(index.KNOWN_PATHS):
        with open(index.KNOWN_PATHS, 'w') as file:
            file.write(json.dumps([], indent=4))


def pytest_sessionfinish(session, exitstatus):
    """Remove tmp config directory again."""
    shutil.rmtree(index.CONFIG_DIR)
