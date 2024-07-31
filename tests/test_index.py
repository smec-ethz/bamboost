# test index.py

import os
import shutil
import sqlite3

import numpy as np
import pytest

from bamboost.index import DatabaseTable, IndexAPI
from bamboost.manager import Manager

from test_manager import temp_manager

# --------------------------------------------------------------------------
# Index creation and database addition (SQLite)
# The index.Index is rerouted to a temporary file -> empty at the beginning
# --------------------------------------------------------------------------
def test_if_index_created():
    assert os.path.isfile(IndexAPI().file)


def test_if_table_created():
    conn = sqlite3.connect(IndexAPI().file, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    # check if table dbindex exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='dbindex';"
    )
    assert cursor.fetchone() is not None


def test_if_database_added(temp_manager: Manager):
    # check if database is in index
    index = IndexAPI().read_table()  # returns pd.DataFrame
    assert temp_manager.UID in index.id.values
    assert temp_manager.path == index.loc[index.id == temp_manager.UID].path.values[0]


def test_if_database_removed(temp_manager: Manager):
    # check if database is removed from index if deleted
    shutil.rmtree(temp_manager.path)
    index = IndexAPI().clean().read_table()
    assert temp_manager.UID not in index.id.values


def test_index_clean(temp_manager: Manager):
    # check if index is cleaned
    shutil.rmtree(temp_manager.path)
    IndexAPI().clean()
    index = IndexAPI().read_table()
    assert index.empty


def test_index_clean_purge(temp_manager: Manager):
    # check if index is cleaned and purged (=table & entry in index is deleted)
    # shutil.rmtree(temp_manager.path)
    uid = temp_manager.UID
    table_name = f"db_{uid}"
    res = IndexAPI().fetch("SELECT name FROM sqlite_master WHERE type='table';")
    # assert that the database table exists
    assert (table_name,) in res

    shutil.rmtree(temp_manager.path)
    IndexAPI().clean(purge=True)
    res = IndexAPI().fetch("SELECT name FROM sqlite_master WHERE type='table';")
    assert f"db_{temp_manager.UID}" not in res


def test_getitem(temp_manager: Manager):
    # check if __getitem__ works -> returns index.DatabaseTable
    dbt = IndexAPI()[temp_manager.UID]
    assert isinstance(dbt, DatabaseTable)


def test_get_id_from_path(temp_manager: Manager):
    # check if get_id_from_path works
    assert IndexAPI().get_id(temp_manager.path) == temp_manager.UID


# --------------------------------------------------------------------------
# Correct transformation and retrieval of data types when writing to SQLite
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", [True, False, np.True_, np.False_])
def test_datatype_parsing_bool(temp_manager: Manager, value: bool):
    # check if boolean is parsed correctly
    sim = temp_manager.create_simulation(parameters={"test": value})
    series = DatabaseTable(temp_manager.UID).read_entry(sim.uid)
    assert series.test == value


@pytest.mark.parametrize("value", [1, 1.0])
def test_datatype_parsing_int_float(temp_manager: Manager, value):
    # check if datatype is parsed correctly
    sim = temp_manager.create_simulation(parameters={"test": value})
    series = DatabaseTable(temp_manager.UID).read_entry(sim.uid)
    assert type(series.test) == type(value)


@pytest.mark.parametrize("array", [np.array([[1, 2, 3], [4.0, 5.0, 6.0]])])
def test_datatype_parsing_array(temp_manager: Manager, array):
    # check if array is parsed correctly
    sim = temp_manager.create_simulation(parameters={"test": array})
    series = DatabaseTable(temp_manager.UID).read_entry(sim.uid)
    assert isinstance(series.test, np.ndarray)


def test_nested_dict(temp_manager: Manager):
    # check if nested dict is parsed correctly (dots are replaced by literal 'DOT' and replaced back when reading)
    sim = temp_manager.create_simulation(
        parameters={"test": {"a": 1, "b": 2, "c": {"d": np.array([2, 3, 4])}}}
    )
    series = DatabaseTable(temp_manager.UID).read_entry(sim.uid)
    assert series["test.a"] == 1
    assert series["test.b"] == 2
    assert np.array_equal(series["test.c.d"], np.array([2, 3, 4]))


def test_read_database_table(temp_manager: Manager):
    # check if database table is read correctly
    temp_manager.create_simulation(
        parameters={"test": True, "false": False, "int": 1, "float": 1.0}
    )
    table = DatabaseTable(temp_manager.UID).read_table()
    assert table.shape[0] == 1
