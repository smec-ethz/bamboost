# This file is part of dbmanager, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from functools import wraps
from abc import ABC, abstractmethod
import time
from typing import Any
import h5py
import logging
from dataclasses import dataclass

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
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self._file:
                log.debug(f"Closing {self._file.filename}")
                self._file.close()

            self.open(mode)
            log.debug(f"Opened {self._file.filename}")

            res = method(self, *args, **kwargs)

            log.debug(f"Closing {self._file.filename}")
            self.close()

            return res

        return wrapper

    return decorator


class FileHandler(ABC):

    def __init__(self, file_name: str) -> None:
        self._file = None
        self.file_name = file_name

    def open(self, mode: str = 'r'):
        self._file = open_h5file(self.file_name, mode)
        return self._file

    def close(self):
        self._file.close()
        

class H5Dataset(FileHandler):
    """Wrapper of a h5py dataset. Storing the file and the infile path. Thus, each call
    opens the file, does operation and closes the file again.

    Args:
        file (`str`): path to h5 file
        path_to_data (`str`): infile path to dataset
    """
    def __init__(self, file_name: str, path_to_data: str) -> None:
        super().__init__(file_name)
        self.path_to_data = path_to_data

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            with self.open('r'):
                return self._file[self.path_to_data].__getattribute__(__name)

    @with_file_open()
    def __repr__(self) -> str:
        return f'{self.__class__} pointing to: ' + self._file[self.path_to_data].__repr__()

    @with_file_open()
    def __str__(self) -> str:
        return self._file[self.path_to_data].__str__()

    @with_file_open('r')
    def __getitem__(self, key):
        value = self._file[self.path_to_data].__getitem__(key)
        if isinstance(value, h5py._hl.base.HLObject):
            return self.__class__(self.file_name, f'{self.path_to_data}/{key}')
        else:
            return value

    @with_file_open('a')
    def __setitem__(self, slice, newvalue):
        self._file[self.path_to_data].__setitem__(slice, newvalue)

    @property
    @with_file_open()
    def shape(self):
        return self._file[self.path_to_data].shape 

    @property
    @with_file_open()
    def dtype(self):
        return self._file[self.path_to_data].dtype
