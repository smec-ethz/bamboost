"""
This module handles the detection and selection of the appropriate MPI implementation for
Bamboost, either using `mpi4py` for real MPI environments or falling back to a mock MPI
implementation for non-MPI environments.

The detection logic considers user configuration options, environment variables, and the
presence of common MPI-related environment variables.

If `mpi4py` is unavailable in an MPI environment, a fallback to the mock implementation is
also provided.

Usage:
    Instead of importing `mpi4py.MPI` directly, import `bamboost.mpi.MPI` to use the
    appropriate MPI module based on the current environment.

    >>> from bamboost.mpi import MPI

Attributes:
    MPI_ON (bool): Flag indicating whether MPI is detected and enabled.
    MPI (module): The selected MPI module (`mpi4py.MPI` or `bamboost.mpi.mock`).
    log (Logger): Logger instance for this module.

Type Aliases:
    Comm: Union of real and mock MPI communicators, available under TYPE_CHECKING.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

from bamboost import BAMBOOST_LOGGER, config

if TYPE_CHECKING:
    from mpi4py.MPI import Comm as _MPIComm

    from bamboost.mpi.mock import Comm as _MockComm

    Comm: TypeAlias = Union[_MPIComm, _MockComm]


log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


def _detect_if_mpi_needed() -> bool:
    if not config.options.mpi:  # user has disabled MPI via config
        return False
    if os.environ.get("BAMBOOST_MPI", None) == "0":  # user has disabled MPI via env
        return False

    # Check if any of the common MPI environment variables are set
    # fmt: off
    mpi_env_vars = {
        "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",        # Open MPI
        "PMI_SIZE", "PMI_RANK",                                # MPICH and Intel MPI
        "MV2_COMM_WORLD_SIZE", "MV2_COMM_WORLD_RANK",          # MVAPICH
        "I_MPI_RANK", "I_MPI_SIZE",                            # Intel MPI
        "SLURM_PROCID", "SLURM_NTASKS",                        # SLURM
        "MPI_LOCALNRANKS", "MPI_LOCALRANKID"                   # General/Other
    }
    # fmt: on
    if mpi_env_vars.intersection(os.environ):
        return True

    log.info("This script does not seem to be run with MPI. Using the mock MPI module.")
    return False


def _get_mpi_module():
    if not MPI_ON:
        import bamboost.mpi.mock as MockMPI

        return MockMPI

    try:
        import mpi4py.MPI as MPI

        return MPI
    except ImportError:
        import bamboost.mpi.mock as MockMPI

        log.info("`mpi4py` unavailable [using the mock MPI module]")
        return MockMPI


MPI_ON = _detect_if_mpi_needed()
MPI = _get_mpi_module()


class Communicator:
    _active_comm = MPI.COMM_WORLD

    def __set__(self, instance, value):
        Communicator._active_comm = value

    def __get__(self, instance, owner) -> Comm:
        return Communicator._active_comm

    def __delete__(self, instance):
        raise AttributeError("Cannot delete the communicator.")
