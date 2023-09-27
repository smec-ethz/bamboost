# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

from typing import Any
import h5py
import numpy as np
import logging

from .file_handler import FileHandler, with_file_open

log = logging.getLogger(__name__)



def get_best_pointer(value: Any) -> BasePointer:
    """Returns pointer based on h5py object type."""
    if isinstance(value, h5py.Group):
        return Group
    if isinstance(value, h5py.Dataset):
        return Dataset
    if isinstance(value, h5py._hl.base.HLObject):
        return BasePointer
    else:
        return None


class BasePointer:
    """Pointer to a location in an hdf5 file. The constructor takes a
    :class:`~.file_handler.FileHandler` and the in-file path to the object.
    The base class represents a generic group in the file

    Args:
        file_handler: file this belongs to
        path_to_data: infile path to object
    """

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        self._file = file_handler
        self.path_to_data = path_to_data

    @property
    def obj(self):
        """The object this BasePointer points to. File needs to be open for access."""
        return self._file[self.path_to_data]

    @with_file_open('r')
    def __str__(self) -> str:
        try:
            return f'{self.__class__} pointing to: ' + self._file[self.path_to_data].__repr__()
        except KeyError:
            return f'{self.__class__} pointing to: {self.path_to_data} (No data at that location!)'

    __repr__ = __str__

    @with_file_open('r')
    def __getattr__(self, __name: str) -> Any:
        """Any attribute request is sent to the h5py object the pointer points to."""
        if hasattr(self.obj, __name):
            return self.obj.__getattribute__(__name)
        return self.__getattribute__(__name)

    @with_file_open('r')
    def __getitem__(self, key):
        value = self.obj[key]
        new_pointer = get_best_pointer(value)
        if new_pointer is None:
            return value
        return new_pointer(self._file, f'{self.path_to_data}/{key}')

    @with_file_open('a')
    def __setitem__(self, slice, newvalue):
        self.obj.__setitem__(slice, newvalue)



class Group(BasePointer):

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)

    def _ipython_key_completions_(self):
        return self.keys()

    @with_file_open('r')
    def keys(self) -> set:
        return set(self.obj.keys())


class MutableGroup(Group):
    """Used for the `userdata` group."""

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)
        # Create group if it doesn't exist
        with self._file('a', driver='mpio'):
            self._file.file_object.require_group(path_to_data)

    @with_file_open('r')
    def __getitem__(self, key) -> Any:
        """Used to access datasets (:class:`~bamboost.common.hdf_pointer.Dataset`)
        or groups inside this group (:class:`~bamboost.common.hdf_pointer.MutableGroup`)
        """
        if isinstance(self.obj[key], h5py.Group):
            return MutableGroup(self._file, f'{self.path_to_data}/{key}')
        return super().__getitem__(key)

    @with_file_open('a')
    def update_attrs(self, attrs: dict) -> None:
        """Update the attributes of the group.

        Args:
            attrs: the dictionary to write as attributes
        """
        self.obj.attrs.update(attrs)

    def add_dataset(self, name: str, vector: np.ndarray, attrs: dict = None) -> None:
        """Add a dataset to the group. Error is thrown if attempting to overwrite
        with different shape than before. If same shape, data is overwritten
        (this is inherited from h5py -> require_dataset)

        Args:
            name: Name for the dataset
            vector: Data to write (max 2d)
            attrs: Optional. Attributes of dataset.
        """
        if attrs is None:
            attrs = {}
        length_local = vector.shape[0]
        length_p = np.array(self._file._comm.allgather(length_local))
        length = np.sum(length_p)
        dim = vector.shape[1:]
        vec_shape = length, *dim

        ranks = np.array([i for i in range(self._file._comm.size)])
        idx_start = np.sum(length_p[ranks<self._file._comm.rank])
        idx_end = idx_start + length_local

        with self._file('a', driver='mpio'):
            dataset = self.obj.require_dataset(name, shape=vec_shape, dtype='f')
            dataset[idx_start:idx_end] = vector
            for key, item in attrs.items():
                dataset.attrs[key] = item
            dataset.flush()

        log.info(f'Written {name} as userdata to {self._file.file_name}...')

    @with_file_open('a')
    def require_group(self, name: str) -> Group:
        """Add a new group to the current group. If exists, return existing.

        Returns:
            :class:`~bamboost.hdf_pointer.Group`
        """
        return MutableGroup(self._file, f'{self.path_to_data}/{name}')


class Dataset(BasePointer):

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)

    @property
    @with_file_open()
    def attrs(self):
        return dict(self.obj.attrs)

    @property
    @with_file_open()
    def shape(self):
        return self.obj.shape 

    @property
    @with_file_open()
    def dtype(self):
        return self.obj.dtype


