import os

import numpy as np
import pytest
from test_manager import temp_manager


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


@pytest.mark.parametrize(
    "data",
    [
        np.array([1, 2, 3]),
        np.array([1.2, 2.3, 3.4]),
        np.array([[1, 2], [3, 4]]),
    ],
)
def test_read_field(temp_manager, data):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.add_field("test", data)
    sim.finish_step()

    sim = temp_manager.sim(uid)
    data_read = sim.data["test"].at_step(0)
    assert np.array_equal(data_read, data)
