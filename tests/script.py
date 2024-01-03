import argparse
import time

import numpy as np
from mpi4py import MPI

from bamboost import SimulationWriter


def run_test(path: str, uid: str) -> None:
    """run the test."""
    sim = SimulationWriter(uid, path)
    array_size = sim.parameters["array_size"]

    # Start time
    start = time.time()
    start_p = MPI.Wtime()

    sim.add_field("test", np.random.randn(array_size))

    # End time
    end = time.time()
    end_p = MPI.Wtime()

    # Store time
    sim.userdata["write_time_total"] = end - start
    mpi_data = sim.userdata.require_group("MPI")
    mpi_data[f"{MPI.COMM_WORLD.rank}"] = end_p - start_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--uid", type=str)
    args = parser.parse_args()
    run_test(args.path, args.uid)
