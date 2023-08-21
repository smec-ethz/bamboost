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
import logging

from .file_handler import FileHandler, with_file_open

log = logging.getLogger(__name__)



class BasePointer: pass
class Group(BasePointer): pass
class Dataset(BasePointer): pass


def get_best_pointer(value: h5py._hl.base.HLObject) -> BasePointer:
    """Returns pointer based on h5py object type."""
    if isinstance(value, h5py.Group):
        return Group
    if isinstance(value, h5py.Dataset):
        return Dataset
    if isinstance(value, h5py._hl.base.HLObject):
        return BasePointer
    else:
        return value  # probably no h5py type object, return it as is


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
        try:  # Test if pointer is valid
            with self._file():
                self.obj
        except KeyError:
            raise KeyError(f"{self.path_to_data} is not a valid location!")

    @property
    def obj(self):
        """The object this BasePointer points to. File needs to be open for access."""
        return self._file[self.path_to_data]

    @with_file_open('r')
    def __str__(self) -> str:
        return f'{self.__class__} pointing to: ' + self._file[self.path_to_data].__repr__()

    __repr__ = __str__

    @with_file_open('r')
    def __getattr__(self, __name: str) -> Any:
        if hasattr(self.obj, __name):
            return self.obj.__getattribute__(__name)
        return self.__getattribute__(__name)

    @with_file_open('r')
    def __getitem__(self, key):
        value = self.obj[key]
        return get_best_pointer(value)

    @with_file_open('a')
    def __setitem__(self, slice, newvalue):
        self.obj.__setitem__(slice, newvalue)



class Group(BasePointer):

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)
        with self._file('r'):
            self._keys = tuple(self.keys())

    def _ipython_key_completions_(self):
        return self._keys

    @with_file_open('r')
    def keys(self):
        return set(self.obj.keys())


class Dataset(BasePointer):

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)

    @property
    @with_file_open()
    def attrs(self):
        return self.obj.attrs

    @property
    @with_file_open()
    def shape(self):
        return self.obj.shape 

    @property
    @with_file_open()
    def dtype(self):
        return self.obj.dtype



