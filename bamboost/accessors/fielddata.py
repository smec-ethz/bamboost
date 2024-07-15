# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

from typing import Generator, Tuple

import h5py
import numpy as np
import pandas as pd

from bamboost import BAMBOOST_LOGGER
from bamboost.accessors.meshes import Mesh, MeshGroup
from bamboost.common import hdf_pointer
from bamboost.common.file_handler import FileHandler, with_file_open

__all__ = ["DataGroup", "FieldData"]

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


# -----------------------------------------------------------------------------


class DataGroup(hdf_pointer.Group):
    """This pointer points to the data directory. Item accessor returns the
    individual data fields. `meshes` is passed to here for access of linked
    meshes.
    """

    def __init__(
        self,
        file_handler: FileHandler,
        meshes: MeshGroup,
        path_to_data: str = "/data",
        **kwargs,
    ) -> None:
        super().__init__(file_handler, path_to_data, **kwargs)
        self.meshes = meshes

    def __getitem__(self, key) -> FieldData:
        return FieldData(self._file, f"{self.path_to_data}/{key}", meshes=self.meshes)

    def __iter__(self) -> Generator[FieldData]:
        for key in self.keys():
            yield self.__getitem__(key)

    @property
    @with_file_open("r")
    def info(self) -> pd.DataFrame:
        """View the data stored.

        Returns:
            :class:`pd.DataFrame`
        """
        tmp_dictionary = dict()
        for data in self:
            steps = len(data)
            shape = data.obj["0"].shape
            dtype = data.obj["0"].dtype
            tmp_dictionary[data._name] = {
                "dtype": dtype,
                "shape": shape,
                "steps": steps,
            }
        return pd.DataFrame.from_dict(tmp_dictionary)


class FieldData(hdf_pointer.Group):
    """This pointer points to a specific data field. `meshes` is passed to here
    for access of linked meshes.
    """

    _vds_key = "__vds"  # We create a virtual dataset for slicing across all steps
    _times_key = (
        "__times"  # We store the time of steps in a seperate dataset for performance
    )

    def __init__(
        self, file_handler: FileHandler, path_to_data: str, meshes: MeshGroup
    ) -> None:
        super().__init__(file_handler, path_to_data)
        self.meshes = meshes
        self._name = path_to_data.split("/")[-1]

    @with_file_open("r")
    def __getitem__(self, key) -> np.ndarray:
        return self._get_full_data()[key]

    def _get_full_data(self) -> h5py.Dataset:
        """Return a HDF5 virtual dataset including all steps of the field.
        
        If the virtual dataset exists with the correct shape it will be
        returned, otherwise it will be created.
        """
        if self._vds_key not in self.keys():
            self._create_vds()
            return self.obj[self._vds_key]

        if len(self) > self.obj.attrs.get("virtual_dataset_length", 0):
            self._create_vds()
            self._create_times()
            return self.obj[self._vds_key]

        return self.obj[self._vds_key]

    @with_file_open("r")
    def __len__(self) -> int:
        return len(self.datasets())

    @with_file_open("r")
    def datasets(self) -> set:
        return super().datasets() - {self._vds_key, self._times_key}

    @property
    @with_file_open("r")
    def shape(self) -> tuple:
        return self._get_full_data().shape

    @property
    @with_file_open("r")
    def dtype(self) -> type:
        return self._get_full_data().dtype

    @with_file_open()
    def at_step(self, *steps: int) -> np.ndarray:
        """Direct access to data at step. Does not require the virtual dataset.

        Args:
            step0, step1, ...: step to extract (can be multiple)
        Returns:
            :class:`np.ndarray`
        """
        data = list()
        for step in steps:
            if step < 0:
                step = len(self) + step
            data.append(self.obj[str(step)][()])
        if len(data) <= 1:
            return data[0]
        else:
            return data

    @property
    @with_file_open()
    def times(self) -> np.ndarray:
        """Return the array of timestamps.

        Returns:
            :class:`np.ndarray`
        """
        if self._times_key not in self.keys():
            self._create_times()
        return self.obj[self._times_key][()]

    @property
    @with_file_open()
    def mesh(self) -> Mesh:
        """Return the linked mesh. Currently returns the linked mesh of first step only.

        Returns:
            :class:`tuple[np.ndarray, np.ndarray]`
        """
        mesh_name = self.obj["0"].attrs.get("mesh", self.meshes._default_mesh)
        return self.meshes[mesh_name]

    @property
    def coordinates(self) -> np.ndarray:
        """Wrapper for mesh.coordinates"""
        return self.mesh.coordinates

    @property
    def connectivity(self) -> np.ndarray:
        """Wrapper for mesh.connectivity"""
        return self.mesh.connectivity

    @property
    @with_file_open()
    def msh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper to get mesh as tuple"""
        return (self.mesh.coordinates, self.mesh.connectivity)

    def regenerate_virtual_datasets(self) -> None:
        """Regenerate virtual dataset. Call this if the data has changed, thus the
        virtual datasets need to be updated to cover the actual data.
        """
        self._create_times()
        self._create_vds()

    @with_file_open("r+")
    def _create_times(self) -> None:
        times = [self.obj[str(step)].attrs.get("t", step) for step in range(len(self))]
        if self._times_key in self.obj.keys():
            del self.obj[self._times_key]
        self.obj.create_dataset(self._times_key, data=np.array(times))

    def _create_vds(self) -> None:
        """Create virtual dataset of the full timeseries of a field.
        Requires HDF5 > 1.10 !
        """
        with self._file("r"):
            length = len(self)
            ds_shape = self.obj["0"].shape
            ds_dtype = self.obj["0"].dtype

            layout = h5py.VirtualLayout(shape=(length, *ds_shape), dtype=ds_dtype)

            for step in range(length):
                vsource = h5py.VirtualSource(self.obj[str(step)])
                layout[step] = vsource

        with self._file("r+"):
            if self._vds_key in self.obj.keys():
                del self.obj[self._vds_key]
            self.obj.attrs["virtual_dataset_length"] = length
            self.obj.create_virtual_dataset(self._vds_key, layout)
