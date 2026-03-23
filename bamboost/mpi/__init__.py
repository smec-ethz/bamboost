"""
This module handles the detection and selection of the appropriate MPI implementation for
Bamboost, either using `mpi4py` for real MPI environments or using the built-in serial
backend for non-MPI environments.

The detection logic considers user configuration options, environment variables, and the
presence of common MPI-related environment variables.

If MPI is detected (via environment variables) but `mpi4py` is not installed, a
:exc:`RuntimeError` is raised immediately rather than silently falling back to a
single-rank fake communicator, which would be unsafe in a real multi-rank launch.

Usage:
    Instead of importing `mpi4py.MPI` directly, import `bamboost.mpi.MPI` to use the
    appropriate MPI module based on the current environment.

    >>> from bamboost.mpi import MPI

Attributes:
    MPI_ON (bool): Flag indicating whether MPI is detected and enabled.
    MPI (module): The selected MPI module (``mpi4py.MPI`` or ``bamboost.mpi.serial``).
    COMM_WORLD: Active world communicator (real or serial).
    COMM_SELF: Per-process communicator (real or serial).
    COMM_NULL: Null communicator sentinel (raises on use).
    log (Logger): Logger instance for this module.

Type Aliases:
    Comm: Union of real and serial MPI communicators, available under TYPE_CHECKING.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Union

from typing_extensions import TypeAlias

from bamboost._config import config
from bamboost._logger import BAMBOOST_LOGGER

if TYPE_CHECKING:
    from mpi4py.MPI import Comm as _MPIComm  # ty: ignore[unresolved-import]

    from bamboost.mpi.serial import SerialComm as _SerialComm

    Comm: TypeAlias = Union[_MPIComm, _SerialComm]


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

    log.info("This script does not seem to be run with MPI. Using the serial backend.")
    return False


def _get_mpi_module() -> tuple[object, bool]:
    """Attempt to import the real MPI module (``mpi4py.MPI``).

    Returns the module and a flag indicating whether real MPI is active.

    Raises:
        RuntimeError: If MPI is required (env vars present) but ``mpi4py`` is not
            installed.  Falling back silently would be unsafe in a multi-rank launch
            because every rank would believe it is rank 0.
    """
    try:
        import mpi4py.MPI as MPI  # ty: ignore[unresolved-import]

        return MPI, True
    except ImportError:
        raise RuntimeError(
            "MPI launch detected (MPI environment variables are set) but `mpi4py` is "
            "not installed.  Install mpi4py (`pip install mpi4py`) or explicitly "
            "disable MPI via `BAMBOOST_MPI=0` or `config.options.mpi = False`."
        ) from None


def _assert_h5py_has_mpi_support() -> None:
    import h5py

    if not h5py.get_config().mpi:
        raise RuntimeError(
            "h5py was not built with MPI support, but MPI is required/enabled in bamboost."
            "Set `config.options.mpi = False` to disable MPI support in bamboost."
        )


def get_mpi_from_env() -> tuple[Any, bool]:
    """Get the MPI module and flag based on environment detection."""
    mpi_needed = _detect_if_mpi_needed()
    if not mpi_needed:
        import bamboost.mpi.serial as SerialMPI

        return SerialMPI, False
    else:
        _assert_h5py_has_mpi_support()  # raises if h5py lacks MPI support
        return _get_mpi_module()


MPI, MPI_ON = get_mpi_from_env()

# Expose communicator constants at the package level so that user code and internal
# modules can import them directly without going through `MPI.*`.
COMM_WORLD = MPI.COMM_WORLD  # ty: ignore[unresolved-attribute]
COMM_SELF = MPI.COMM_SELF  # ty: ignore[unresolved-attribute]
COMM_NULL = MPI.COMM_NULL  # ty: ignore[unresolved-attribute]


class Communicator:
    _active_comm: Comm = COMM_WORLD  # ty: ignore[unresolved-attribute]

    def __set__(self, instance, value):
        Communicator._active_comm = value

    def __get__(self, instance, owner) -> Comm:
        return Communicator._active_comm

    def __delete__(self, instance):
        raise AttributeError("Cannot delete the communicator.")
