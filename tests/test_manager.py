# test_manager.py

import os
import shutil
import tempfile

import pytest

from bamboost import Manager, Simulation, index


@pytest.fixture()
def temp_manager():
    temp_dir = tempfile.mkdtemp()
    db = Manager(path=temp_dir)
    yield db
    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        pass


def test_if_new_manager_created():
    temp_dir = tempfile.mkdtemp()
    db = Manager(temp_dir)
    assert os.path.isdir(temp_dir)


class TestManager:

    def test_create_simulation_creates_new_file(self, temp_manager: Manager):
        sim_uid = "this"
        writer = temp_manager.create_simulation(uid=sim_uid)
        h5_file = os.path.join(temp_manager.path, sim_uid, f"{sim_uid}.h5")
        assert os.path.isfile(h5_file)

    @pytest.mark.parametrize("fix_df", [True, False])
    def test_manager_length(self, temp_manager: Manager, fix_df: bool):
        temp_manager.FIX_DF = fix_df
        temp_manager.create_simulation()
        temp_manager.create_simulation()
        temp_manager.create_simulation()
        assert len(temp_manager) == 3

    def test_get_single_simulation_base_case(self, temp_manager: Manager):
        sim_uid = "this"
        temp_manager.create_simulation()
        temp_manager.create_simulation(sim_uid)
        db = Manager(temp_manager.path)
        sim = db.sim(sim_uid)
        assert isinstance(sim, Simulation)
        assert sim.path == os.path.join(temp_manager.path, sim_uid)


# --------------------------------------------------------------------------
# Database syncing between SQLite and Filesystem
# --------------------------------------------------------------------------
def test_if_table_created(temp_manager: Manager):
    uid = temp_manager.UID
    table_name = f"db_{uid}"
    res = index.Index.fetch("SELECT name FROM sqlite_master WHERE type='table';")
    # assert that the database table exists
    assert (table_name,) in res
    # assert that the table with update times exists
    assert (f"db_{uid}_t",) in res

