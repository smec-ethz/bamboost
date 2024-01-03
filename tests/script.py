import argparse
import time

import h5py
import numpy as np
from mpi4py import MPI

from bamboost import SimulationWriter


def run_test(path: str, uid: str) -> None:
    """run the test."""
    sim = SimulationWriter(uid, path)
    array_size = sim.parameters["array_size"]
    nb_processes = sim.parameters["nb_processes"]
    array_size_per_process = array_size // nb_processes

    # Start time
    start = time.time()
    start_p = MPI.Wtime()

    sim.add_field("test", np.random.randn(array_size_per_process, array_size))

    # End time
    end = time.time()
    end_p = MPI.Wtime()

    sim.add_global_field("time", end - start)

    # Store time
    sim.userdata["write_time_total"] = end - start
    mpi_data = sim.userdata.require_group("MPI")
    with sim.open("a", driver="mpio") as f:
        obj: h5py._h5py.Group = mpi_data.obj
        ds = obj.require_dataset("times", (MPI.COMM_WORLD.Get_size(),), float)
        ds[MPI.COMM_WORLD.Get_rank()] = end_p - start_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--uid", type=str)
    args = parser.parse_args()
    run_test(args.path, args.uid)

