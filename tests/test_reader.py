import os

from test_manager import temp_manager


class TestReader:

    def test_some(self):
        assert 1 == 1


def test_enter_path(temp_manager):
    sim = temp_manager.create_simulation()
    sim.finish_sim()
    path = sim.path
    uid = sim.uid
    original_path = os.getcwd()
    with temp_manager.sim(uid).enter_path():
        assert os.getcwd() == path
    assert os.getcwd() == original_path

    # check that we return in the right path even when throwing an error
    try:
        with temp_manager.sim(uid).enter_path():
            raise ValueError()
    except ValueError:
        assert os.getcwd() == original_path
