# This file is part of dbmanager, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

from functools import wraps
from abc import ABC, abstractmethod
import time
from typing import Any, Union
import h5py
import logging

log = logging.getLogger(__name__)

HAS_MPIO = 'mpio' in h5py.registered_drivers()
if HAS_MPIO:
    MPI_ACTIVE = h5py._MPI_ACTIVE
else:
    MPI_ACTIVE = False


def open_h5file(file: str, mode, driver=None, comm=None):
    """Open h5 file. Waiting if file is not available.

    Args:
        file (str): File to open
        mode (str): 'r', 'a', 'w', ...
        driver (str): driver for h5.File
        comm: MPI communicator
    """
    while True:
        try:
            if driver=='mpio' and HAS_MPIO and MPI_ACTIVE:
                return h5py.File(file, mode, driver=driver, comm=comm)
            else:
                return h5py.File(file, mode)

        except OSError:
            log.info(f"File {file} not accessible, waiting") 
            time.sleep(1)


def with_file_open(mode: str = 'r'):
    """Open the file (`self._file`) before function
    Close the file after the function call

    Works on classes containing the member `_file` of type :class:`~dbmanager.common.file_handler.FileHandler`
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            with self._file(mode):
                return method(self, *args, **kwargs)
        return wrapper
    return decorator


class FileHandler:
    """File handler for an hdf5 file with the purpose of handling opening and closing
    of the file. We use the concept of composition to include an object of this type
    in classes which need access to an hdf5 file (such as the hdf5pointer and Simulation.)
    """
    def __init__(self, file_name: str) -> None:
        self.file_object: h5py.File = None
        self.file_name = file_name
        self._lock = 0
        self._mode = 'r'

    def __call__(self, mode: str = 'r') -> FileHandler:
        self._mode = mode
        return self

    def __enter__(self):
        self.open(self._mode)
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key) -> Any:
        return self.file_object[key]

    def __getattr__(self, __name: str) -> Any:
        return self.file_object.__getattribute__(__name)

    def open(self, mode: str = 'r'):
        if self._lock==0:
            log.debug(f"[{id(self)}] Open {self.file_name}")
            self.file_object = open_h5file(self.file_name, mode)
        self._lock += 1
        log.debug(f"[{id(self)}] Lock count {self._lock}")
        log.debug(f"h5py id = {self.file_object.id}")
        return self.file_object

    def close(self):
        self._lock -= 1
        if self._lock==0:
            log.debug(f"[{id(self)}] Close {self.file_name}")
            self.file_object.close()
        log.debug(f"[{id(self)}] Lock count {self._lock}")

    def change_file_mode(self, mode: str):
        log.debug(f"Forced closing and reopening to change file mode [{self.file_name}]")
        self.file_object.close()
        self.file_object = open_h5file(self.file_name, mode)
        

class HDF5Pointer:
    """Wrapper of a h5py dataset. Storing the file and the infile path. Thus, each call
    opens the file, does operation and closes the file again.

    Args:
        file (`str`): path to h5 file
        path_to_data (`str`): infile path to dataset
    """

    @property
    def obj(self):
        if self._attribute:
            return self._file[self.path_to_data].__getattribute__(self._attribute)
        else:
            return self._file[self.path_to_data]

    def __init__(self, file_handler: FileHandler, path_to_data: str, *, _attribute: str = None) -> None:
        self._file = file_handler
        self.path_to_data = path_to_data
        self._attribute = _attribute

    @with_file_open('r')
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.obj.__call__(*args, **kwargs)

    @with_file_open('r')
    def __getattr__(self, __name: str) -> Any:
        if hasattr(self.obj, __name):
            return self.obj.__getattribute__(__name)
        return self.__getattribute__(__name)

    @with_file_open()
    def __repr__(self) -> str:
        return f'{self.__class__} pointing to: ' + self._file[self.path_to_data].__repr__()

    @with_file_open()
    def __str__(self) -> str:
        return f'{self.__class__} pointing to: ' + self._file[self.path_to_data].__repr__()

    @with_file_open('r')
    def __getitem__(self, key):
        value = self.obj.__getitem__(key)
        if isinstance(value, h5py._hl.base.HLObject):
            return self.__class__(self._file, f'{self.path_to_data}/{key}')
        else:
            return value

    @with_file_open('a')
    def __setitem__(self, slice, newvalue):
        self.obj.__setitem__(slice, newvalue)

    @with_file_open()
    def keys(self):
        return set(self.obj.keys())

    @property
    @with_file_open()
    def shape(self):
        return self.obj.shape 

    @property
    @with_file_open()
    def dtype(self):
        return self.obj.dtype
