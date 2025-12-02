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
    assert MPI.__name__ == "bamboost.mpi.mock"  # type: ignore
