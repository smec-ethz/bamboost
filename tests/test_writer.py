import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from test_manager import temp_manager


@pytest.mark.parametrize(
    "dtype,array",
    [
        ["<f4", np.array([1, 2], dtype=int)],
        ["int32", np.array([1, 2])],
    ],
)
def test_field_dtype_forced(temp_manager, dtype, array):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.add_field("test", array, dtype=dtype)

    sim.finish_step()
    sim.add_field("test", 2 * array, dtype=dtype)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    assert sim.data["test"].dtype == dtype


@pytest.mark.parametrize(
    "dtype,value",
    [
        ["<f4", 1],
        ["int32", 1],
    ],
)
def test_global_field_dtype_forced(temp_manager, dtype, value):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.add_global_field("test", value, dtype=dtype)

    sim.finish_step()
    sim.add_global_field("test", 2 * value, dtype=dtype)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    assert sim.globals["test"].dtype == dtype


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.0, 2.0]),
        np.array([1, 2], dtype="int"),
    ],
)
def test_field_dtype_inferred(temp_manager, array):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.add_field("test", array)

    sim.finish_step()
    sim.add_field("test", 2 * array)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    # The dtypes are equivalent (although they may have another string representation)
    assert np.dtype(sim.data["test"].dtype) == array.dtype


@pytest.mark.parametrize(
    "val",
    [
        1.0,
        1,
    ],
)
def test_global_field_dtype_inferred(temp_manager, val):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.add_global_field("test", val)

    sim.finish_step()
    sim.add_global_field("test", 2 * val)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    # The dtypes are equivalent (although they may have another string representation)
    assert np.dtype(sim.globals["test"].dtype) == np.array([val]).dtype


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.0, 2.0]),
        np.array([1, 2], dtype="int"),
    ],
)
def test_userdata_dtype_inferred(temp_manager, array):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.userdata.add_dataset("test", array)

    sim.finish_step()
    sim.userdata.add_dataset("test", 2 * array)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    # The dtypes are equivalent (although they may have another string representation)
    assert np.dtype(sim.userdata["test"].dtype) == array.dtype


@pytest.mark.parametrize(
    "dtype,array",
    [
        ["<f4", np.array([1, 2], dtype=int)],
        ["int32", np.array([1, 2])],
    ],
)
def test_userdata_dtype_forced(temp_manager, dtype, array):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.userdata.add_dataset("test", array, dtype=dtype)

    sim.finish_step()
    sim.userdata.add_dataset("test", 2 * array, dtype=dtype)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    assert sim.userdata["test"].dtype == dtype


def test_enter_path(temp_manager):
    sim = temp_manager.create_simulation()
    sim.finish_sim()
    path = sim.path
    uid = sim.uid
    original_path = os.getcwd()
    with sim.enter_path():
        assert os.getcwd() == path
    assert os.getcwd() == original_path

    # check that we return in the right path even when throwing an error
    try:
        with sim.enter_path():
            raise ValueError()
    except ValueError:
        assert os.getcwd() == original_path


@pytest.mark.parametrize(
    "array",
    [
        np.array([1, 2, 3]),
        np.array([[1, 2, 3], [2, 3, 4]]),
        np.array([1]),
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    ],
)
def test_add_field_dimensions(temp_manager, array):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.add_field("test", array)

    sim.finish_step()
    sim.add_field("test", 2 * array)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    np.testing.assert_array_equal(sim.data["test"].at_step(0), array)
    np.testing.assert_array_equal(sim.data["test"].at_step(1), 2 * array)


def test_add_fields(temp_manager):
    # Mock the add_field method
    simulation_writer = temp_manager.create_simulation()

    # Test data
    fields = {
        "field1": np.array([1, 2, 3]),
        "field2": (np.array([4, 5, 6]), "Cell"),
        "field3": np.array([7, 8, 9]),
    }
    time = 1.0
    mesh = "test_mesh"

    # Call the method
    simulation_writer.add_fields(fields, time, mesh)

    assert simulation_writer.data.keys() == fields.keys()
    assert simulation_writer.data["field2/0"].attrs["center"] == "Cell"
    assert simulation_writer.data["field1/0"].attrs["center"] == "Node"
    np.testing.assert_array_equal(
        simulation_writer.data["field3"][0], np.array([7, 8, 9])
    )
