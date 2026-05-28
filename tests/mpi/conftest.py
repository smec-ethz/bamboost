"""Pytest tests for MPI/parallel execution in Bamboost.

Needs:
- mpi4py
- h5py built against mpi
    export CC=mpicc
    export HDF5_MPI="ON"
    uv pip install --no-binary=h5py h5py
"""

import sys

import pytest


def pytest_collection_modifyitems(config, items):
    """Automatically mark all tests in this directory with the 'mpi' marker."""
    for item in items:
        if "tests/mpi" in str(item.fspath):
            item.add_marker(pytest.mark.mpi)


@pytest.fixture(autouse=True)
def verify_mpi_available():
    """Verify that MPI is truly active for tests in this directory."""
    import importlib.util

    mpi_available = importlib.util.find_spec("mpi4py") is not None

    if not mpi_available:
        pytest.fail(
            "MPI is not active, but tests in `tests/mpi/` require a real MPI environment. "
            "Ensure you are running with `mpirun` or `mpiexec` and that `mpi4py` is installed."
        )


@pytest.fixture(autouse=True)
def mpi_barrier_synchronizer():
    """Ensure all MPI ranks are perfectly synchronized before and after each test.

    This prevents ranks from drifting apart or entering race conditions in fixture setup.
    """
    from bamboost.mpi import MPI

    if MPI is not None and hasattr(MPI, "COMM_WORLD"):
        MPI.COMM_WORLD.barrier()
        yield
        try:
            MPI.COMM_WORLD.barrier()
        except Exception:
            # If a rank aborted/failed during the test, ignore exceptions in teardown
            pass
    else:
        yield


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Abort all MPI processes immediately if any rank encounters a failure.

    This prevents standard pytest runs from hanging in deadlocks when one process fails
    while others continue to wait on collective operations (like barriers or bcast).
    """
    if call.excinfo is not None:
        # Ignore skipped tests and expected outcomes
        if call.excinfo.typename in ("Skipped", "Skip", "Exception"):
            # Double check if it is a pytest skip
            if call.excinfo.errisinstance(pytest.skip.Exception):
                return

        from bamboost.mpi import MPI

        if MPI is not None and hasattr(MPI, "COMM_WORLD") and MPI.COMM_WORLD.size > 1:
            sys.stderr.write(
                f"\n\n=================== [MPI Rank {MPI.COMM_WORLD.rank}] TEST FAILURE DETECTED ===================\n"
                f"Aborting all processes immediately to prevent MPI deadlocks...\n"
                f"Exception: {call.excinfo}\n"
                f"=================================================================================\n\n"
            )
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)
