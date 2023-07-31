# This file is part of dbmanager, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

from typing import Any, Tuple, Union
from types import SimpleNamespace
import numpy as np
import h5py
import logging
import pandas as pd
from mpi4py import MPI

from .simulation import Simulation
from .common.file_handler import with_file_open, HDF5Pointer

log = logging.getLogger(__name__)

class Group: pass
class Data(Group): pass
class Field(Group): pass
class SimulationReader(Simulation): pass


class Group:

    def __init__(self, _group: HDF5Pointer, sim: Simulation, _name: str) -> None:
        self._group = _group
        self._file = self._group._file
        self._name = _name
        self.sim = sim
        self.uid = sim.uid

    def __str__(self) -> str:
        return f"{self.__class__} {self._name} in {self.uid}"

    __repr__ = __str__

    @with_file_open('r')
    def __getitem__(self, key) -> Field:
        return Group(HDF5Pointer(self._file, f'{self._group.name}/{key}'), self.sim, key)

    def __getattr__(self, attr) -> Any:
        if hasattr(self._group, attr):
            return self._group.__getattribute__(attr)
        else:
            return self.__getattribute__(attr)
    

class Data(Group):

    def __init__(self, _group: HDF5Pointer, sim: Simulation, _name: str) -> None:
        super().__init__(_group, sim, _name)

        with self._file():
            self._getitem_keys = list(self._group.keys())

    @with_file_open('r')
    def __getitem__(self, key) -> Field:
        return Field(HDF5Pointer(self._file, f'{self._group.name}/{key}'), self.sim, key)
    
    def _ipython_key_completions_(self):
        return self._getitem_keys


class MeshGroup(Group):
    
    def __init__(self, _group: HDF5Pointer, sim: Simulation, _name: str) -> None:
        super().__init__(_group, sim, _name)
        with self._file():
            self._getitem_keys = list(self._group.keys())

    @with_file_open('r')
    def __getitem__(self, key) -> Mesh:
        return Mesh(HDF5Pointer(self._file, f'{self._group.name}/{key}'), self.sim, key)

    def _ipython_key_completions_(self):
        return self._getitem_keys


class Mesh(Group):

    def __init__(self, _group: HDF5Pointer, sim: Simulation, _name: str) -> None:
        super().__init__(_group, sim, _name)

    @property
    @with_file_open('r')
    def coordinates(self):
        return self._group['geometry'][()]

    @property
    @with_file_open('r')
    def connectivity(self):
        return self._group['topology'][()]


class Field(Group):

    _vds_key = '__vds'
    _times_key = '__times'

    def __init__(self, _group: HDF5Pointer, sim: Simulation, _name: str) -> None:
        super().__init__(_group, sim, _name)

    @with_file_open('r')
    def __getitem__(self, key) -> Any:
        return self._get_full_data()[key]

    def __len__(self) -> int:
        count = 0
        reduntants = (self._vds_key, self._times_key)
        with self._file:
            for key in reduntants:
                count += 1 if key in self._group.obj.keys() else 0
            return len(self._group.obj.keys()) - count

    @property
    def times(self) -> HDF5Pointer:
        """Return the array of timestamps.

        Returns:
            :class:`~dbmanager.common.file_handler.HDF5Pointer`
        """
        try:
            return self._group[self._times_key]
        except KeyError:
            self._create_times()
            return self._group[self._times_key]

    @property
    @with_file_open()
    def mesh(self) -> Mesh:
        """Return the linked mesh. Currently returns the linked mesh of first step only.

        Returns:
            :class:`tuple[np.ndarray, np.ndarray]`
        """
        mesh_name = self._group['0'].attrs.get('mesh', self.sim._default_mesh)
        return self.sim.meshes[mesh_name]
         
    @property
    @with_file_open()
    def msh(self) -> Tuple[np.ndarray, np.ndarray]:
        mesh_name = self._group['0'].attrs.get('mesh', self.sim._default_mesh)
        return self.sim.get_mesh(mesh_name)

    @with_file_open()
    def at_step(self, *steps: int) -> np.ndarray:
        """Direct access to data at step. Does not require the virtual dataset.

        Args:
            step (`int`): step to extract
        Returns:
            :class:`np.ndarray`
        """
        data = list()
        for step in steps:
            if step<0:
                step = len(self) + step
            data.append(self._group[str(step)][()])
        if len(data)<=1:
            return data[0]
        else:
            return data

    def regenerate_virtual_datasets(self) -> None:
        """Regenerate virtual dataset. Call this if the data has changed, thus the
        virtual datasets need to be updated to cover the actual data.
        """
        self._create_times()
        self._create_vds()

    def _get_full_data(self) -> HDF5Pointer:
        try:
            return self._group['__vds']
        except KeyError:
            self._create_vds()
            return self._group['__vds']

    @with_file_open()
    def _create_times(self) -> None:
        if self._times_key not in self._group.obj.keys():
            times = list()
            for step in range(len(self)):
                times.append(self._group.obj[str(step)].attrs.get('t', step))

            self._file.change_file_mode('a')
            if self._times_key in self._group.obj.keys():
                del self._group.obj[self._times_key]
            self._group.obj.create_dataset(self._times_key, data=np.array(times))
            self._file.change_file_mode('r')


    def _create_vds(self) -> None:
        """Create virtual dataset of the full timeseries of a field.
        Requires HDF5 > 1.10 !
        """

        with self._file:

            grp = self._group.obj
            length = len(self)
            ds_shape = grp['0'].shape
            ds_dtype = grp['0'].dtype

            layout = h5py.VirtualLayout(shape=(length, *ds_shape), dtype=ds_dtype)
            
            for step in range(length):
                vsource = h5py.VirtualSource(grp[str(step)])
                layout[step] = vsource

        with self._file:
            self._file.change_file_mode('a')
            if self._vds_key in self._group.obj.keys():
                del self._group.obj[self._vds_key]

            self._group.obj.create_virtual_dataset(self._vds_key, layout)
            self._file.change_file_mode('r')


# ---------------------------------------------------------------------------------------


class SimulationReader(Simulation):

    def __init__(self, uid: str, path: str, comm: MPI.Comm = MPI.COMM_WORLD):
        super().__init__(uid, path, comm)

        try:
            self.data: Data = Data(HDF5Pointer(self._file, 'data'), sim=self, _name='data')
        except KeyError:
            log.warning(f"Data not found in this simulation file.")
            self.data = "File doesn't contain data"

        try:
            self.meshes: MeshGroup = MeshGroup(HDF5Pointer(self._file, self._mesh_location), self, 'Mesh')
        except KeyError:
            log.warning("MeshGroup not found in this simulation file.")
            self.meshes = "File doesn't contain mesh data"
        
    @with_file_open()
    def print_hdf5_file_structure(self, print_datasets: bool = False):
        """Print data in file."""
        def print_hdf5_item(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset) and print_datasets:
                print(f"Dataset: {name}")
        self._file.visititems(print_hdf5_item)

    @property
    def mesh(self):
        """Return coordinates and connectivity of default mesh."""
        return self.get_mesh()

    @with_file_open('r')
    def get_mesh(self, mesh_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return coordinates and connectivity. Currently returns numpy arrays.

        Args:
            mesh_name (`str`): optional, name of mesh to read (default = mesh)
        """
        if mesh_name is None:
            mesh_name = self._default_mesh

        mesh = self.meshes[mesh_name]
        return mesh.coordinates, mesh.connectivity

    @property
    @with_file_open('r')
    def globals(self):
        """Return globals."""
        grp = self._file['globals']
        d = {key: grp[key][()] for key in grp.keys()}
        return pd.DataFrame.from_dict(d)

    @property
    @with_file_open('r')
    def data_info(self) -> pd.Dataframe:
        """View the data stored."""
        tmp_dictionary = dict()
        for data in self._file['data'].keys():
            steps = max([int(step) for step in self._file[f'data/{data}'].keys()]) + 1
            shape = self._file[f'data/{data}/0'].shape
            dtype = self._file[f'data/{data}/0'].dtype
            tmp_dictionary[data] = {'dtype': dtype, 'shape': shape, 'steps': steps}
        return pd.DataFrame.from_dict(tmp_dictionary)

    @property
    @with_file_open('r')
    def git(self) -> dict:
        """Get Git information.

        Returns:
            Dictionary with different repositories.
        """
        if 'git' not in self._file.keys():
            return "Sorrrry, no git information stored :()"
        grp = self._file['git']
        tmp_dict = {}
        for repo in grp.keys():
            tmp_dict[repo] = grp[repo][()].decode('utf8')
        return tmp_dict

    @property
    def log(self) -> str:
        """Get exception traceback of failed simulations"""
        with self._file('r') as f:
            return f['log'][()].decode('utf8')

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

    def get_data_interpolator(self, name: str, step: int):
        """Get Linear interpolator for data field.

        Args:
            name: name of the data field
            step: step
        """
        from scipy.interpolate import LinearNDInterpolator
        return LinearNDInterpolator(self.mesh[0], self.data[name].at_step(step))



