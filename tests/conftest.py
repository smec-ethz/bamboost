import os
import shutil
import tempfile
from pathlib import Path

from bamboost import config

config.options.mpi = False


def pytest_sessionstart(session):
    """Setup tmp config directory."""
    tempdir = tempfile.mkdtemp()
    tempdir = Path(tempdir)
    config.paths.localDir = tempdir
    config.paths.cacheDir = tempdir.joinpath("cache")
    config.index.databaseFile = tempdir.joinpath("index.sqlite")

    # Create config files if they don't exist
    os.makedirs(config.paths.localDir, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    """Remove tmp config directory again."""
    shutil.rmtree(config.paths.localDir)
