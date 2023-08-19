# This file is part of dbmanager, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

from typing import Any, Tuple
import numpy as np
import h5py
import logging

from .simulation import Simulation
from .common.file_handler import with_file_open, HDF5Pointer

log = logging.getLogger(__name__)



class BaseH5Object: pass
class GroupFieldData(BaseH5Object): pass
class GroupMeshes(BaseH5Object): pass
class FieldData(BaseH5Object): pass
class Mesh(BaseH5Object): pass


class BaseH5Object:
    """Base object in h5 file structure. Usually this is a group.

    Args:
        - pointer: Pointer to this object
        - sim: Simulation object this belongs to (not nice that this is needed)
        - _name: name for object (usually last part of in-file path)
    """

    def __init__(self, pointer: HDF5Pointer, sim: Simulation, _name: str) -> None:
        self.pointer = pointer
        self._file = self.pointer._file
        self._name = _name
        self.sim = sim
        self.uid = sim.uid

    def __str__(self) -> str:
        return f"{self.__class__} {self._name} in {self.uid}"

    __repr__ = __str__

    @with_file_open('r')
    def __getitem__(self, key) -> FieldData:
        return BaseH5Object(HDF5Pointer(self._file, f'{self.pointer.name}/{key}'), self.sim, key)

    def __getattr__(self, attr) -> Any:
        if hasattr(self.pointer, attr):
            return self.pointer.__getattribute__(attr)
        else:
            return self.__getattribute__(attr)
    

class GroupFieldData(BaseH5Object):

    def __init__(self, pointer: HDF5Pointer, sim: Simulation, _name: str) -> None:
        super().__init__(pointer, sim, _name)

        with self._file():
            self.fields = list(self.pointer.keys())

    def __iter__(self) -> FieldData:
        for field in self.fields:
            yield FieldData(HDF5Pointer(self._file, f'{self.pointer.name}/{field}'), self.sim, field)

    @with_file_open('r')
    def __getitem__(self, key) -> FieldData:
        return FieldData(HDF5Pointer(self._file, f'{self.pointer.name}/{key}'), self.sim, key)
    
    def _ipython_key_completions_(self):
        return self.fields


class FieldData(BaseH5Object):

    _vds_key = '__vds'
    _times_key = '__times'

    def __init__(self, pointer: HDF5Pointer, sim: Simulation, _name: str) -> None:
        super().__init__(pointer, sim, _name)

    @with_file_open('r')
    def __getitem__(self, key) -> np.ndarray:
        return self._get_full_data()[key]

    @with_file_open('r')
    def __len__(self) -> int:
        count = 0
        reduntants = (self._vds_key, self._times_key)
        for key in reduntants:
            count += 1 if key in self.pointer.obj.keys() else 0
        return len(self.pointer.obj.keys()) - count

    @property
    def shape(self) -> tuple:
        return self._get_full_data().shape

    @property
    def coordinates(self) -> np.ndarray:
        return self.mesh.coordinates

    @property
    def connectivity(self) -> np.ndarray:
        return self.mesh.connectivity

    @property
    def times(self) -> HDF5Pointer:
        """Return the array of timestamps.

        Returns:
            :class:`~dbmanager.common.file_handler.HDF5Pointer`
        """
        try:
            # try to access the times dataset in the object of the pointer
            return self.pointer[self._times_key]
        except KeyError:
            self._create_times()
            return self.pointer[self._times_key]

    @property
    @with_file_open()
    def mesh(self) -> Mesh:
        """Return the linked mesh. Currently returns the linked mesh of first step only.

        Returns:
            :class:`tuple[np.ndarray, np.ndarray]`
        """
        mesh_name = self.pointer['0'].attrs.get('mesh', self.sim._default_mesh)
        return self.sim.meshes[mesh_name]
         
    @property
    @with_file_open()
    def msh(self) -> Tuple[np.ndarray, np.ndarray]:
        mesh_name = self.pointer['0'].attrs.get('mesh', self.sim._default_mesh)
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
            data.append(self.pointer[str(step)][()])
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
            return self.pointer['__vds']
        except KeyError:
            self._create_vds()
            return self.pointer['__vds']

    @with_file_open('r+')
    def _create_times(self) -> None:
        times = list()
        for step in range(len(self)):
            times.append(self.pointer.obj[str(step)].attrs.get('t', step))

        if self._times_key in self.pointer.obj.keys():
            del self.pointer.obj[self._times_key]
        self.pointer.obj.create_dataset(self._times_key, data=np.array(times))


    def _create_vds(self) -> None:
        """Create virtual dataset of the full timeseries of a field.
        Requires HDF5 > 1.10 !
        """

        with self._file('r'):

            grp = self.pointer.obj
            length = len(self)
            ds_shape = grp['0'].shape
            ds_dtype = grp['0'].dtype

            layout = h5py.VirtualLayout(shape=(length, *ds_shape), dtype=ds_dtype)
            
            for step in range(length):
                vsource = h5py.VirtualSource(grp[str(step)])
                layout[step] = vsource

        with self._file('r+'):
            if self._vds_key in self.pointer.obj.keys():
                del self.pointer.obj[self._vds_key]

            self.pointer.obj.create_virtual_dataset(self._vds_key, layout)


class GroupMeshes(BaseH5Object):
    
    def __init__(self, pointer: HDF5Pointer, sim: Simulation, _name: str) -> None:
        super().__init__(pointer, sim, _name)
        with self._file():
            self.fields = list(self.pointer.keys())

    @with_file_open('r')
    def __getitem__(self, key) -> Mesh:
        return Mesh(HDF5Pointer(self._file, f'{self.pointer.name}/{key}'), self.sim, key)

    def _ipython_key_completions_(self):
        return self.fields


class Mesh(BaseH5Object):

    def __init__(self, pointer: HDF5Pointer, sim: Simulation, _name: str) -> None:
        super().__init__(pointer, sim, _name)

    @property
    @with_file_open('r')
    def coordinates(self):
        return self.pointer['geometry'][()]

    @property
    @with_file_open('r')
    def connectivity(self):
        return self.pointer['topology'][()]
