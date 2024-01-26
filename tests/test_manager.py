# test_manager.py

from bamboost import Manager, Simulation
import os
import tempfile
import shutil
import pytest


@pytest.fixture()
def temp_manager():
    temp_dir = tempfile.mkdtemp()
    db = Manager(path=temp_dir)
    print(db.df)
    yield db
    shutil.rmtree(temp_dir)


class TestManager:

    def test_if_new_manager_created(self):
        temp_dir = tempfile.mkdtemp()
        db = Manager(temp_dir)
        assert os.path.isdir(temp_dir)

    def test_create_simulation_creates_new_file(self, temp_manager: Manager):
        sim_uid = 'this'
        writer = temp_manager.create_simulation(uid=sim_uid)
        h5_file = os.path.join(temp_manager.path, sim_uid, f'{sim_uid}.h5')
        assert os.path.isfile(h5_file)

    def test_manager_length(self, temp_manager: Manager):
        temp_manager.create_simulation()
        temp_manager.create_simulation()
        assert len(temp_manager)==2

    def test_get_single_simulation_base_case(self, temp_manager: Manager):
        sim_uid = 'this'
        temp_manager.create_simulation()
        temp_manager.create_simulation(sim_uid)
        db = Manager(temp_manager.path)
        sim = db.sim(sim_uid)
        assert isinstance(sim, Simulation)
        assert sim.path == os.path.join(temp_manager.path, sim_uid)

