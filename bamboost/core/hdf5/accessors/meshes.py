# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

from typing import Tuple

import numpy as np

from bamboost import BAMBOOST_LOGGER
from bamboost.core.hdf5.file import FileMode, Group, HDF5File, with_file_open

__all__ = ["MeshGroup", "Mesh"]

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


class MeshGroup(Group):
    def __init__(
        self,
        file_handler: HDF5File,
        path_to_data: str = "/Mesh/0",
        _default_mesh: str = "mesh",
        **kwargs,
    ) -> None:
        super().__init__(path_to_data, file_handler, **kwargs)
        self._default_mesh = _default_mesh

    @with_file_open(FileMode.READ)
    def __getitem__(self, key) -> Mesh:
        return Mesh(self._file, f"{self._name}/{key}")


class Mesh(Group):
    def __init__(self, file_handler: HDF5File, path_to_data: str) -> None:
        super().__init__(path_to_data, file_handler)

    @property
    @with_file_open(FileMode.READ)
    def coordinates(self) -> np.ndarray:
        try:
            return self._obj["geometry"][()]  # type: ignore
        except KeyError:
            return self._obj["coordinates"][()]  # type: ignore

    @property
    @with_file_open(FileMode.READ)
    def connectivity(self) -> np.ndarray:
        return self._obj["topology"][()]  # type: ignore

    @with_file_open(FileMode.READ)
    def get_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.coordinates, self.connectivity)
