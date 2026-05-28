import sys
from pathlib import Path

import numpy as np
import pytest

from bamboost import Collection, Simulation, config
from bamboost.core.hdf5.file import HDF_MPI_ACTIVE
from bamboost.mpi import MPI
from bamboost.mpi.utilities import RootProcessMeta, comm_self


@pytest.fixture
def mpi_collection(tmp_path_factory: pytest.TempPathFactory):
    """Isolated, MPI-safe collection fixture that explicitly synchronizes across ranks."""
    comm = MPI.COMM_WORLD

    # Convert Path to string on Rank 0 before broadcasting to avoid any pickling issues
    if comm.rank == 0:
        path_str = str(tmp_path_factory.mktemp("mpi_test"))
    else:
        path_str = None

    shared_path_str = comm.bcast(path_str, root=0)
    shared_path = Path(shared_path_str)

    # Ensure Rank 0 creates the parent directory first, then barrier
    if comm.rank == 0:
        shared_path.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    # Instantiate the collection, explicitly passing the parallel communicator
    coll = Collection(
        path=shared_path,
        comm=comm,
    )
    yield coll


@pytest.mark.mpi(min_size=2)
def test_collection_parallel_init(mpi_collection: Collection):
    """Verify that collection path and UID are properly synchronized across ranks."""
    comm = mpi_collection._comm

    # Gather UIDs and paths from all processes
    all_uids = comm.allgather(mpi_collection.uid)
    all_paths = comm.allgather(str(mpi_collection.path))

    # All ranks must have the identical UID and path
    assert len(set(all_uids)) == 1, f"Collection UIDs diverged: {all_uids}"
    assert len(set(all_paths)) == 1, f"Collection paths diverged: {all_paths}"


@pytest.mark.mpi(min_size=2)
def test_simulation_parallel_add(mpi_collection: Collection):
    """Verify that adding a simulation with generated name remains synchronized."""
    comm = mpi_collection._comm

    # Add a simulation without specifying a name on any rank
    sim = mpi_collection.add(parameters={"parallel_test": True, "rank": comm.rank})

    # Gather simulation names from all ranks
    all_names = comm.allgather(sim.name)
    all_uids = comm.allgather(str(sim.uid))

    # Assert names and UIDs generated are identical across ranks
    assert len(set(all_names)) == 1, f"Simulation names diverged: {all_names}"
    assert len(set(all_uids)) == 1, f"Simulation UIDs diverged: {all_uids}"


@pytest.mark.mpi(min_size=2)
def test_comm_self_propagation(mpi_collection: Collection):
    """Verify that changing sim._comm via comm_self also changes sim._file._comm."""
    sim = mpi_collection.add("test_comm_self")

    # Initially they should be COMM_WORLD (or the mpi_collection comm)
    comm = mpi_collection._comm
    assert sim._comm == comm
    assert sim._file._comm == comm

    with comm_self(sim):
        # Now they should be COMM_SELF
        assert sim._comm == MPI.COMM_SELF
        assert sim._file._comm == MPI.COMM_SELF

    # After context they should be back to original
    assert sim._comm == comm
    assert sim._file._comm == comm


@pytest.mark.skipif(
    not HDF_MPI_ACTIVE, reason="h5py was not compiled with parallel MPIO support"
)
@pytest.mark.mpi(min_size=2)
def test_parallel_dataset_write(mpi_collection: Collection):
    """Test parallel collective dataset writing using add_numerical_dataset."""
    comm = mpi_collection._comm

    # Add a dedicated simulation for parallel dataset writes
    sim = mpi_collection.add(
        "test_parallel_dataset",
        override=True,
    )

    # Define process-local data chunk
    # Rank 0 writes [10.0], Rank 1 writes [11.0], etc.
    local_val = float(comm.rank + 10)
    local_data = np.array([local_val], dtype=np.float64)

    # Write the dataset collectively in parallel
    with sim.edit() as writer:
        writer.root.write_distributed_contiguous_array("parallel_vec", local_data)

    # All ranks synchronize after writing
    comm.barrier()

    # Read back the dataset on all ranks and assert correctness
    with sim.open("r") as reader:
        dataset = reader["parallel_vec"][()]
        expected = np.array([float(r + 10) for r in range(comm.size)], dtype=np.float64)
        assert np.allclose(dataset, expected), (
            f"Dataset did not match expected parallel write. Got {dataset}, expected {expected}"
        )


@pytest.mark.mpi(min_size=2)
def test_explicit_comm_when_globally_disabled(tmp_path_factory, monkeypatch):
    """Verify that even if globally MPI is disabled, passing an explicit comm works."""
    from mpi4py import MPI as real_MPI

    comm = real_MPI.COMM_WORLD

    # monkeypatch global MPI settings to simulate globally disabled MPI
    monkeypatch.setattr(MPI, "enabled", False)
    from bamboost.mpi import serial

    monkeypatch.setattr(MPI, "_mpi_module", serial)

    # Convert Path to string on Rank 0 before broadcasting to avoid any pickling issues
    if comm.rank == 0:
        path_str = str(tmp_path_factory.mktemp("mpi_disabled_test"))
    else:
        path_str = None

    shared_path_str = comm.bcast(path_str, root=0)
    shared_path = Path(shared_path_str)

    # Ensure Rank 0 creates the parent directory first, then barrier
    if comm.rank == 0:
        shared_path.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    # Instantiate the collection, explicitly passing the parallel communicator
    coll = Collection(
        path=shared_path,
        comm=comm,
    )

    # 1. Verify UID and path are synchronized
    all_uids = comm.allgather(coll.uid)
    assert len(set(all_uids)) == 1

    # 2. Verify adding a simulation in parallel works
    sim = coll.add(parameters={"parallel_disabled_test": True, "rank": comm.rank})

    all_names = comm.allgather(sim.name)
    assert len(set(all_names)) == 1

    # 3. Verify editing & writing parallel dataset works if HDF_MPI_ACTIVE
    local_val = float(comm.rank + 100)
    local_data = np.array([local_val], dtype=np.float64)

    with sim.edit() as writer:
        writer.root.write_distributed_contiguous_array("parallel_vec", local_data)

    comm.barrier()

    with sim.open("r") as reader:
        dataset = reader["parallel_vec"][()]
        expected = np.array(
            [float(r + 100) for r in range(comm.size)], dtype=np.float64
        )
        assert np.allclose(dataset, expected)
