# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2024 Flavio Lorez and contributors
#
# There is no warranty for this code
import os
from typing import Any, Union

from bamboost import BAMBOOST_LOGGER
from bamboost._config import config
from bamboost.common._mock_mpi import MockMPI

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])

MPIType = Union[MockMPI, Any]

MPI_ON = config.get("options", {}).get("mpi", True)
ENV_BAMBOOST_MPI: bool = os.environ.get("BAMBOOST_MPI", None)
"""Indicates the use of `mpi4py.MPI`. If `0`, the `MockMPI` class is used
instead. Is set by reading the environment variable `BAMBOOST_MPI` [0 or 1].
"""
if ENV_BAMBOOST_MPI is not None:
    MPI_ON = ENV_BAMBOOST_MPI == "1"


def _get_mpi_module():
    if not MPI_ON:
        return MockMPI

    try:
        from mpi4py import MPI

        return MPI
    except ImportError:
        log.info("`mpi4py` unavailable [using a mock MPI module]")
        return MockMPI


MPI: Union[MockMPI, Any] = _get_mpi_module()
