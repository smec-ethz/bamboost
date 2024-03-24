# test index.py

import os
import shutil
import sqlite3
import tempfile

import pytest
from test_manager import temp_manager

from bamboost.common import sqlite
from bamboost.index import DatabaseTable, Index
from bamboost.manager import Manager


@pytest.fixture()
def temp_database():
    """Create a temporary sqlite database for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Index
    shutil.rmtree(temp_dir)


def test_if_db_created():
    assert os.path.isfile(Index.file)


def test_if_table_created():
    conn = sqlite3.connect(Index.file, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    # check if table dbindex exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='dbindex';"
    )
    assert cursor.fetchone() is not None


def test_if_database_added(temp_manager: Manager):
    # check if database is in index
    index = Index.read_table()  # returns pd.DataFrame
    assert temp_manager.UID in index.id.values
    assert temp_manager.path == index.loc[index.id == temp_manager.UID].path.values[0]


@pytest.mark.parametrize("value", [True, False])
def test_datatype_parsing_bool(temp_manager: Manager, value: bool):
    # check if boolean is parsed correctly
    sim = temp_manager.create_simulation(parameters={"test": value})
    series = DatabaseTable(temp_manager.UID).read_entry(sim.uid)
    assert type(series.test) == bool


@pytest.mark.parametrize("value", [1, 1.0])
def test_datatype_parsing_int_float(temp_manager: Manager, value):
    # check if datatype is parsed correctly
    sim = temp_manager.create_simulation(parameters={"test": value})
    series = DatabaseTable(temp_manager.UID).read_entry(sim.uid)
    assert type(series.test) == type(value)


def test_read_database_table(temp_manager: Manager):
    # check if database table is read correctly
    temp_manager.create_simulation(
        parameters={"test": True, "false": False, "int": 1, "float": 1.0}
    )
    table = DatabaseTable(temp_manager.UID).read_table()
    print(table)
    assert table.shape[0] == 1
