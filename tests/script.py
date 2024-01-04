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
    nb_steps = sim.parameters["nb_steps"]
    array_size_per_process = array_size // nb_processes
    dim = 3

    # Start time
    sim._comm.Barrier()
    start = MPI.Wtime()

    with sim:
        for _ in range(nb_steps):
            sim.add_field("test", np.random.randn(array_size_per_process, dim))
            sim.finish_step()

    # End time
    sim._comm.Barrier()
    end = MPI.Wtime()

    # Store time
    sim.userdata["write_time_total"] = end - start
    # mpi_data = sim.userdata.require_group("MPI")
    # with sim.open("a", driver="mpio"):
    #     obj: h5py._h5py.Group = mpi_data.obj
    #     ds = obj.require_dataset("times", (MPI.COMM_WORLD.Get_size(),), float)
    #     ds[MPI.COMM_WORLD.Get_rank()] = end_p - start_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--uid", type=str)
    args = parser.parse_args()
    run_test(args.path, args.uid)

