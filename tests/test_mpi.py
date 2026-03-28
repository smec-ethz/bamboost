import importlib
import os

import pytest
from _pytest.monkeypatch import MonkeyPatch

from bamboost.mpi import (
    _assert_h5py_has_mpi_support,
    _detect_if_mpi_needed,
    get_mpi_from_env,
)


@pytest.fixture
def h5py_mpi_enabled(monkeypatch):
    class FakeConfig:
        mpi = True

    monkeypatch.setattr("h5py.get_config", lambda: FakeConfig())
    yield


@pytest.fixture
def h5py_mpi_disabled(monkeypatch):
    class FakeConfig:
        mpi = False

    monkeypatch.setattr("h5py.get_config", lambda: FakeConfig())
    yield


def test_detect_mpi_disabled_by_config(monkeypatch: MonkeyPatch):
    monkeypatch.setattr("bamboost.config.options.mpi", False)
    assert _detect_if_mpi_needed() is False


def test_detect_mpi_disabled_via_env(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("BAMBOOST_MPI", "0")
    assert _detect_if_mpi_needed() is False


def test_detect_mpi_env_vars(monkeypatch: MonkeyPatch):
    monkeypatch.setattr("bamboost.config.options.mpi", True)
    monkeypatch.delenv("BAMBOOST_MPI", raising=False)
    monkeypatch.setenv("PMI_SIZE", "16")  # one of the common MPI env vars
    assert _detect_if_mpi_needed() is True


def test_detect_no_mpi(monkeypatch: MonkeyPatch):
    monkeypatch.setattr("bamboost.config.options.mpi", True)
    for key in list(os.environ):
        if key.startswith(("OMPI", "PMI", "MV2", "I_MPI", "SLURM", "MPI_")):
            monkeypatch.delenv(key, raising=False)
    assert _detect_if_mpi_needed() is False


def test_h5py_mpi_check(monkeypatch, h5py_mpi_disabled):
    with pytest.raises(RuntimeError, match="h5py was not built with MPI support"):
        _assert_h5py_has_mpi_support()


def test_h5py_mpi_check_on_import(monkeypatch: MonkeyPatch, h5py_mpi_disabled):
    # make sure MPI_ON is triggered to be True
    monkeypatch.setattr("bamboost.config.options.mpi", True)
    monkeypatch.setenv("PMI_SIZE", "16")  # ensure MPI is detected as needed

    with pytest.raises(RuntimeError):
        import bamboost.mpi as mpi

        # reload the mpi module to trigger the check
        # Should raise because h5py.get_config().mpi is False
        importlib.reload(mpi)


def test_mpi_integration_on(monkeypatch, h5py_mpi_enabled):
    monkeypatch.setattr("bamboost.config.options.mpi", True)
    monkeypatch.setenv("PMI_SIZE", "16")  # force MPI detection
    # Mock the _get_mpi_module to return a fake module and True
    # skips actual mpi4py import
    monkeypatch.setattr(
        "bamboost.mpi._get_mpi_module", lambda: ("fake-real-comm", True)
    )

    MPI, MPI_ON = get_mpi_from_env()

    assert MPI == "fake-real-comm"
    assert MPI_ON is True


def test_mpi_integration_off(monkeypatch):
    monkeypatch.setattr("bamboost.config.options.mpi", False)
    MPI, MPI_ON = get_mpi_from_env()

    assert MPI_ON is False
    assert MPI.__name__ == "bamboost.mpi.serial"  # type: ignore


def test_mpi_needed_but_mpi4py_missing(monkeypatch, h5py_mpi_enabled):
    """When MPI env vars are present but mpi4py cannot be imported, raise RuntimeError."""
    monkeypatch.setattr("bamboost.config.options.mpi", True)
    monkeypatch.setenv("PMI_SIZE", "16")  # force MPI detection
    # Simulate mpi4py being unavailable by making the real import fail
    monkeypatch.setattr(
        "bamboost.mpi._get_mpi_module",
        lambda: (_ for _ in ()).throw(
            RuntimeError(
                "MPI launch detected (MPI environment variables are set) but `mpi4py` is "
                "not installed."
            )
        ),
    )

    with pytest.raises(RuntimeError, match="mpi4py"):
        get_mpi_from_env()


def test_serial_comm_basic():
    """SerialComm has rank=0, size=1, bcast is identity, barrier is a no-op."""
    from bamboost.mpi.serial import SerialComm

    comm = SerialComm()
    assert comm.rank == 0
    assert comm.size == 1
    assert comm.bcast(42) == 42
    assert comm.bcast("hello", root=0) == "hello"
    comm.barrier()  # should not raise


def test_serial_comm_multi_rank_ops_raise():
    """Multi-rank operations on SerialComm raise NotImplementedError."""
    from bamboost.mpi.serial import SerialComm

    comm = SerialComm()
    with pytest.raises(NotImplementedError):
        comm.scatter([1, 2, 3])
    with pytest.raises(NotImplementedError):
        comm.gather(1)
    with pytest.raises(NotImplementedError):
        comm.allreduce(1)
    with pytest.raises(NotImplementedError):
        comm.reduce(1)
    with pytest.raises(NotImplementedError):
        comm.send(1, dest=1)
    with pytest.raises(NotImplementedError):
        comm.recv(source=1)


def test_serial_comm_allgather():
    """allgather in serial mode returns a single-element list."""
    from bamboost.mpi.serial import SerialComm

    comm = SerialComm()
    assert comm.allgather(42) == [42]
    assert comm.allgather("hello") == ["hello"]


def test_null_comm_raises_on_use():
    """NullComm raises RuntimeError on any attribute access."""
    from bamboost.mpi.serial import NullComm

    null = NullComm()
    with pytest.raises(RuntimeError, match="COMM_NULL"):
        _ = null.rank


def test_comm_world_and_comm_self_are_distinct():
    """COMM_WORLD and COMM_SELF are separate instances in serial mode."""
    from bamboost.mpi.serial import COMM_SELF, COMM_WORLD

    assert COMM_WORLD is not COMM_SELF


def test_root_process_meta_comm_self_context_manager_serial():
    """RootProcessMeta.comm_self swaps _comm to COMM_SELF and restores it."""
    import bamboost.mpi as mpi
    from bamboost.mpi.utilities import RootProcessMeta

    class FakeObj:
        _comm = mpi.COMM_WORLD

    obj = FakeObj()
    original = obj._comm

    with RootProcessMeta.comm_self(obj):
        assert obj._comm is mpi.COMM_SELF
        assert obj._comm is not original

    assert obj._comm is original
