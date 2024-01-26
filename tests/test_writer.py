from test_manager import temp_manager
import numpy as np
import pytest


@pytest.mark.parametrize("dtype,array", [['<f4', np.array([1, 2], dtype=int)],
                                         ["int32", np.array([1, 2])],
                                         ])
def test_field_dtype_forced(temp_manager, dtype, array):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.add_field("test", array, dtype=dtype)

    sim.finish_step()
    sim.add_field("test", 2 * array, dtype=dtype)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    assert sim.data["test"].dtype == dtype


@pytest.mark.parametrize("dtype,value", [['<f4', 1],
                                         ["int32", 1],
                                         ])
def test_global_field_dtype_forced(temp_manager, dtype, value):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.add_global_field("test", value, dtype=dtype)

    sim.finish_step()
    sim.add_global_field("test", 2 * value, dtype=dtype)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    assert sim.globals["test"].dtype == dtype


@pytest.mark.parametrize("array", [np.array([1., 2.]),
                                   np.array([1, 2], dtype="int"),
                                   ])
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


@pytest.mark.parametrize("val", [1., 1, ])
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


@pytest.mark.parametrize("array", [np.array([1., 2.]),
                                   np.array([1, 2], dtype="int"),
                                   ])
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


@pytest.mark.parametrize("dtype,array", [['<f4', np.array([1, 2], dtype=int)],
                                         ["int32", np.array([1, 2])],
                                         ])
def test_userdata_dtype_forced(temp_manager, dtype, array):
    sim = temp_manager.create_simulation()
    uid = sim.uid
    sim.userdata.add_dataset("test", array, dtype=dtype)

    sim.finish_step()
    sim.userdata.add_dataset("test", 2 * array, dtype=dtype)

    sim.finish_sim()

    sim = temp_manager.sim(uid)

    assert sim.userdata["test"].dtype == dtype
