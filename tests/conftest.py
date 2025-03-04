import os
import shutil
import tempfile
from pathlib import Path

import pytest

from bamboost import Collection, Index, config


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

@pytest.fixture(scope="module")
def tmp_path_module():
    tmp_path = tempfile.mkdtemp()
    yield Path(tmp_path)
    shutil.rmtree(tmp_path)


def _create_tmp_collection():
    temp_dir = tempfile.mkdtemp()
    db = Collection(path=temp_dir, index_instance=Index.default)
    try:
        yield db
    finally:
        try:
            shutil.rmtree(temp_dir)
        except FileNotFoundError:
            pass


@pytest.fixture(scope="module")
def tmp_collection():
    yield from _create_tmp_collection()


@pytest.fixture
def tmp_collection_burn():
    yield from _create_tmp_collection()


@pytest.fixture(scope="module")
def test_collection(tmp_collection: Collection):
    tmp_collection.create_simulation(
        "testsim1",
        parameters={
            "first_name": "John",
            "age": 20,
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
        },
    )
    tmp_collection.create_simulation(
        "testsim2",
        parameters={
            "first_name": "Jane",
            "age": 30,
            "list": [4, 5, 6],
            "dict": {"c": 3, "d": 4},
        },
    )
    tmp_collection.create_simulation(
        "testsim3",
        parameters={
            "first_name": "Jack",
            "age": 40,
            "list": [7, 8, 9],
            "dict": {"e": 5, "f": 6},
        },
    )
    yield tmp_collection
