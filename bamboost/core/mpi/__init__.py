# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2024 Flavio Lorez and contributors
#
# There is no warranty for this code
import os
from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

from bamboost import BAMBOOST_LOGGER, config

if TYPE_CHECKING:
    from mpi4py.MPI import Comm as _MPIComm

    from bamboost.core.mpi.mock import Comm as _MockComm

    Comm: TypeAlias = Union[_MPIComm, _MockComm]


log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


def _detect_if_mpi_needed() -> bool:
    if not config.options.mpi:  # user has disabled MPI via config
        return False
    if os.environ.get("BAMBOOST_MPI", None) == "0":  # user has disabled MPI via env
        return False

    # Check if any of the common MPI environment variables are set
    # fmt: off
    mpi_env_vars = [
        "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",        # Open MPI
        "PMI_SIZE", "PMI_RANK",                                # MPICH and Intel MPI
        "MV2_COMM_WORLD_SIZE", "MV2_COMM_WORLD_RANK",          # MVAPICH
        "I_MPI_RANK", "I_MPI_SIZE",                            # Intel MPI
        "SLURM_PROCID", "SLURM_NTASKS",                        # SLURM
        "MPI_LOCALNRANKS", "MPI_LOCALRANKID"                   # General/Other
    ]
    # fmt: on
    if any(var in os.environ for var in mpi_env_vars):
        return True

    log.info("This script does not seem to be run with MPI. Using the mock MPI module.")
    return False


def _get_mpi_module():
    if not MPI_ON:
        import bamboost.core.mpi.mock as MockMPI

        return MockMPI

    try:
        import mpi4py.MPI as MPI

        return MPI
    except ImportError:
        import bamboost.core.mpi.mock as MockMPI

        log.info("`mpi4py` unavailable [using the mock MPI module]")
        return MockMPI


MPI_ON = _detect_if_mpi_needed()
MPI = _get_mpi_module()
