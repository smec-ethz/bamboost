# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2024 Flavio Lorez and contributors
#
# There is no warranty for this code
import logging
import os
from typing import Any, Union

log = logging.getLogger(__name__)

from ._mock_mpi import MockMPI

MPI_ON: bool = False if os.environ.get("BAMBOOST_NO_MPI", "0") == "1" else True
"""Indicates the use of `mpi4py.MPI`. If `False`, the `MockMPI` class is used
instead. Is set by reading the environment variable `BAMBOOST_NO_MPI` [0 or 1].
"""

def _get_mpi_module():
    if not MPI_ON:
        return MockMPI

    try:
        from mpi4py import MPI
        return MPI
    except ImportError:
        log.warning("MPI is not available, using MockMPI")
        return MockMPI

MPI: Union[MockMPI, Any] = _get_mpi_module()
