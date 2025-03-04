import os
import shutil
import tempfile
from pathlib import Path

import pytest

from bamboost import config


def pytest_sessionstart(session):
    """Setup tmp config directory."""
    # Use tempdir for testing
    tempdir = tempfile.mkdtemp()
    tempdir = Path(tempdir)
    config.paths.localDir = tempdir
    config.paths.cacheDir = tempdir.joinpath("cache")
    # Disable MPI for testing
    config.options.mpi = False
    # Use in-memory database for testing
    config.index.databaseFile = ":memory:"

    # Create config files if they don't exist
    os.makedirs(config.paths.localDir, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    """Remove tmp config directory again."""
    shutil.rmtree(config.paths.localDir)


@pytest.fixture
def tmp_path():
    tmp_path = tempfile.mkdtemp()
    yield Path(tmp_path)
    shutil.rmtree(tmp_path)


@pytest.fixture
def test_collection(tmp_path: Path):
    from bamboost.core.collection import Collection

    shutil.copytree(
        f"{os.path.dirname(__file__)}/test_collection", tmp_path / "test_collection"
    )

    coll = Collection(tmp_path / "test_collection")
    return coll
