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
    array_size_per_process = array_size[0] // nb_processes
    dim = array_size[1]

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

    # Print result:
    if sim._comm.rank == 0:
        print(
            f"Writing {nb_steps} steps of {array_size_per_process}x{dim} "
            f"arrays with {nb_processes} processes took {end - start:.3f} seconds."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--uid", type=str)
    args = parser.parse_args()
    run_test(args.path, args.uid)

