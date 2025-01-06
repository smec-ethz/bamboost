# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np

from bamboost import BAMBOOST_LOGGER
from bamboost.core.simulation.base import Simulation
from bamboost.mpi import MPI

__all__ = ["SimulationWriter"]

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


class SimulationWriter(Simulation):
    """The SimulationWriter is the writer object for a single simulation. It inherits
    all reading methods from :class:`Simulation`.

    This class can be used as a context manager. When entering the context, the status
    of the simulation is changed to "Started". When an exception is raised inside the
    context, the status is changed to "Failed [Exception]".

    Args:
        uid: The identifier of the simulation
        path: The (parent) database path
        comm: An MPI communicator (Default: `MPI.COMM_WORLD`)
    """


    def add_fields(
        self,
        fields: Dict[str, np.ndarray | Tuple[np.ndarray, str]],
        time: float = None,
        mesh: str = None,
    ) -> None:
        """Add multiple fields at once.

        Args:
            fields: Dictionary with fields. The value can be a tuple with the
                data and a string "Node" or "Cell".
            time: Optional. time
        """
        for key, value in fields.items():
            if isinstance(value, tuple):
                vector, center = value
            else:
                vector, center = value, "Node"
            self.add_field(key, vector, time, mesh, center=center)


    def add_global_fields(self, fields: Dict[str, Any]) -> None:
        """Add multiple global fields at once.

        Args:
            fields: Dictionary with fields
        """
        for name, value in fields.items():
            self.add_global_field(name, value)


