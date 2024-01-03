# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from ..common import hdf_pointer
from ..common.file_handler import FileHandler, with_file_open

__all__ = ["MeshGroup", "Mesh"]

log = logging.getLogger(__name__)


class MeshGroup(hdf_pointer.Group):
    def __init__(
        self,
        file_handler: FileHandler,
        path_to_data: str = "/Mesh/0",
        _default_mesh: str = "mesh",
        **kwargs,
    ) -> None:
        super().__init__(file_handler, path_to_data, **kwargs)
        self._default_mesh = _default_mesh

    @with_file_open("r")
    def __getitem__(self, key) -> Mesh:
        return Mesh(self._file, f"{self.path_to_data}/{key}")


class Mesh(hdf_pointer.Group):
    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)

    @property
    @with_file_open("r")
    def coordinates(self):
        return self.obj["geometry"][()]

    @property
    @with_file_open("r")
    def connectivity(self):
        return self.obj["topology"][()]

    @with_file_open("r")
    def get_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.coordinates, self.connectivity)
