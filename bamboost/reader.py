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
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import h5py
from mpi4py import MPI

from .simulation import Simulation
from .common import hdf_pointer
from .common.file_handler import FileHandler, with_file_open

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class DataGroup(hdf_pointer.Group):

    def __init__(self, file_handler: FileHandler, meshes: MeshGroup,
                 path_to_data: str = '/data') -> None:
        super().__init__(file_handler, path_to_data)
        self.meshes = meshes

    def __getitem__(self, key) -> FieldData:
        return FieldData(self._file, f'{self.path_to_data}/{key}', meshes=self.meshes)

    def __iter__(self) -> FieldData:
        for key in self._keys:
            yield self.__getitem__(key)

    @property
    @with_file_open('r')
    def info(self) -> pd.Dataframe:
        """View the data stored.

        Returns:
            :class:`pd.DataFrame`
        """
        tmp_dictionary = dict()
        for data in self:
            steps = len(data)
            shape = data.obj['0'].shape
            dtype = data.obj['0'].dtype
            tmp_dictionary[data._name] = {'dtype': dtype, 'shape': shape, 'steps': steps}
        return pd.DataFrame.from_dict(tmp_dictionary)


class FieldData(hdf_pointer.Group):

    _vds_key = '__vds'
    _times_key = '__times'

    def __init__(self, file_handler: FileHandler, path_to_data: str,
                 meshes: MeshGroup) -> None:
        super().__init__(file_handler, path_to_data)
        self.meshes = meshes
        self._name = path_to_data.split('/')[-1]

    @with_file_open('r')
    def __getitem__(self, key) -> np.ndarray:
        return self._get_full_data()[key]

    def _get_full_data(self) -> h5py.Dataset:
        try:
            return self.obj[self._vds_key]
        except KeyError:
            self._create_vds()
            return self.obj[self._vds_key]

    @with_file_open('r')
    def __len__(self) -> int:
        non_field_keys = (self._vds_key, self._times_key)
        nb_non_field_keys = sum(1 for key in non_field_keys if key in self.keys())
        return len(self.keys()) - nb_non_field_keys

    @property
    @with_file_open('r')
    def shape(self) -> tuple:
        return self._get_full_data().shape

    @property
    @with_file_open('r')
    def dtype(self) -> type:
        return self._get_full_data().dtype

    @with_file_open()
    def at_step(self, *steps: int) -> np.ndarray:
        """Direct access to data at step. Does not require the virtual dataset.

        Args:
            *step: step to extract (can be multiple)
        Returns:
            :class:`np.ndarray`
        """
        data = list()
        for step in steps:
            if step<0:
                step = len(self) + step
            data.append(self.obj[str(step)][()])
        if len(data)<=1:
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
        mesh_name = self.obj['0'].attrs.get('mesh', self.meshes._default_mesh)
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

    @with_file_open('r+')
    def _create_times(self) -> None:
        times = [self.obj[str(step)].attrs.get('t', step)
                 for step in range(len(self))
                 ]
        if self._times_key in self.obj.keys():
            del self.obj[self._times_key]
        self.obj.create_dataset(self._times_key, data=np.array(times))

    def _create_vds(self) -> None:
        """Create virtual dataset of the full timeseries of a field.
        Requires HDF5 > 1.10 !
        """
        with self._file('r'):
            length = len(self)
            ds_shape = self.obj['0'].shape
            ds_dtype = self.obj['0'].dtype

            layout = h5py.VirtualLayout(shape=(length, *ds_shape), dtype=ds_dtype)
            
            for step in range(length):
                vsource = h5py.VirtualSource(self.obj[str(step)])
                layout[step] = vsource

        with self._file('r+'):
            if self._vds_key in self.obj.keys():
                del self.obj[self._vds_key]
            self.obj.create_virtual_dataset(self._vds_key, layout)


class MeshGroup(hdf_pointer.Group):

    def __init__(self, file_handler: FileHandler, path_to_data: str = '/Mesh/0',
                 _default_mesh: str = 'mesh') -> None:
        super().__init__(file_handler, path_to_data)
        self._default_mesh = _default_mesh
    
    @with_file_open('r')
    def __getitem__(self, key) -> Mesh:
        return Mesh(self._file, f'{self.path_to_data}/{key}')


class Mesh(hdf_pointer.Group):

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)

    @property
    @with_file_open('r')
    def coordinates(self):
        return self.obj['geometry'][()]

    @property
    @with_file_open('r')
    def connectivity(self):
        return self.obj['topology'][()]

    @with_file_open('r')
    def get_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.coordinates, self.connectivity)


# -----------------------------------------------------------------------------

class SimulationReader(Simulation):
    """Viewer of a single simulation of the database.

    Args:
        uid (`str`): unique identifier
        path (`str`): database path
        comm (:class:`MPI.COMM_WORLD`): mpi communicator
    """
    meshes: MeshGroup = None
    data: DataGroup = None

    def __init__(self, uid: str, path: str, comm: MPI.Comm = MPI.COMM_WORLD):
        super().__init__(uid, path, comm)

        try:  # Create views of data and mesh if these exist
            self.meshes = MeshGroup(self._file, self._mesh_location, self._default_mesh)
        except KeyError:
            log.warning(f'No mesh data in {self.uid}.')

        try:
            self.data = DataGroup(self._file, self.meshes)
        except KeyError:
            log.warning(f'No field data in {self.uid}.')

    def open(self, mode: str = 'r', driver=None, comm=None) -> FileHandler:
        """Use this as a context manager in a `with` statement.
        Purpose: keeping the file open to directly access/edit something in the
        HDF5 file of this simulation.

        Args:
            mode (`str`): file mode (see h5py docs)
            driver (`str`): file driver (see h5py docs)
            comm (`str`): mpi communicator
        """
        return self._file(mode, driver, comm)
        
    @property
    def mesh(self):
        """Return coordinates and connectivity of default mesh.

        Returns:
            Tuple of np.arrays (coordinates, connectivity)
        """
        return self.get_mesh()

    @with_file_open('r')
    def get_mesh(self, mesh_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return coordinates and connectivity. Currently returns numpy arrays.

        Args:
            mesh_name (`str`): optional, name of mesh to read (default = mesh)
        Returns:
            Tuple of np.arrays (coordinates, connectivity)
        """
        if mesh_name is None:
            mesh_name = self._default_mesh

        mesh = self.meshes[mesh_name]
        return mesh.coordinates, mesh.connectivity

    @property
    @with_file_open('r')
    def globals(self) -> pd.DataFrame:
        """Return global data.

        Returns:
            :class:`pd.DataFrame`
        """
        grp = self._file['globals']
        d = {key: grp[key][()] for key in grp.keys()}
        return pd.DataFrame.from_dict(d)

    @property
    @with_file_open('r')
    def data_info(self) -> pd.Dataframe:
        """View the data stored.

        Returns:
            :class:`pd.DataFrame`
        """
        tmp_dictionary = dict()
        for data in self.data:
            steps = len(data)
            shape = data.obj['0'].shape
            dtype = data.obj['0'].dtype
            tmp_dictionary[data._name] = {'dtype': dtype, 'shape': shape, 'steps': steps}
        return pd.DataFrame.from_dict(tmp_dictionary)

    @property
    @with_file_open('r')
    def git(self) -> dict:
        """Get Git information.

        Returns:
            :class:`dict` with different repositories.
        """
        if 'git' not in self._file.keys():
            return "Sorrrry, no git information stored :()"
        grp = self._file['git']
        tmp_dict = {}
        for repo in grp.keys():
            tmp_dict[repo] = grp[repo][()].decode('utf8')
        return tmp_dict

    @property
    @with_file_open('r')
    def post(self) -> SimpleNamespace:
        """Return the data stored in the postprocess category.

        Returns:
            SimpleNamespace with all data.
        """
        data = self._file['postprocess']
        d = {key: data[key][()] for key in data.keys()}
        return SimpleNamespace(**d)

    def get_data_interpolator(self, field: str, step: int):
        """Get Linear interpolator for data field at step. Uses the linked mesh.

        Args:
            name (`str`): name of the data field
            step (`int`): step
        Returns:
            :class:`scipy.interpolate.LinearNDInterpolator`
        """
        from scipy.interpolate import LinearNDInterpolator
        return LinearNDInterpolator(self.data[field].mesh.coordinates,
                                    self.data[field].at_step(step))

    @with_file_open()
    def print_hdf5_file_structure(self, print_datasets: bool = False):
        """Print data structure in file to screen."""
        def print_hdf5_item(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset) and print_datasets:
                print(f"Dataset: {name}")
        self._file.visititems(print_hdf5_item)




