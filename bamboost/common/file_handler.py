# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import os
import time
from functools import wraps
from typing import Any, Literal, Type

import h5py

from bamboost.common import mpi
from bamboost import BAMBOOST_LOGGER

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])

__all__ = [
    "open_h5file",
    "FileHandler",
    "with_file_open",
    "capture_key_error",
]

HAS_MPIO = "mpio" in h5py.registered_drivers()
if HAS_MPIO and mpi.MPI_ON:
    MPI_ACTIVE = h5py.h5.get_config().mpi
else:
    MPI_ACTIVE = False

FILE_MODE_HIRARCHY = {
    "r": 1,
    "r+": 2,
    "a": 2,
    "w": 3,
}


def open_h5file(
    file: str,
    mode: Literal["mpio"] | Type[None],
    driver: bool | Type[None] = None,
    comm=None,
):
    """Open h5 file. Waiting if file is not available.

    Args:
        file (str): File to open
        mode (str): 'r', 'a', 'w', ...
        driver (str): driver for h5.File
        comm: MPI communicator
    """
    while True:
        try:
            if driver == "mpio" and MPI_ACTIVE and mpi.MPI_ON:
                return h5py.File(file, mode, driver=driver, comm=comm)
            else:
                return h5py.File(file, mode)

        except BlockingIOError:
            log.warning(f"file locked --> {file}")
            time.sleep(0.2)


def with_file_open(mode: str = "r", driver=None, comm=None):
    """Open the file (`self._file`) before function
    Close the file after the function call

    Works on classes containing the member `_file` of type :class:`~bamboost.common.file_handler.FileHandler`
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            with self._file(mode, driver, comm):
                return method(self, *args, **kwargs)

        return wrapper

    return decorator


def capture_key_error(method):
    @wraps(method)
    def inner(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except KeyError as e:
            e.add_note(f"[file: {self.file_name}]")
            raise e

    return inner


class FileHandler:
    """File handler for an hdf5 file with the purpose of handling opening and closing
    of the file. We use the concept of composition to include an object of this type
    in classes which need access to an hdf5 file (such as the hdf5pointer and Simulation.)

    Args:
        file_name: the path to the file

    Attributes:
        file_object: the h5py file object (accessible if open)
        _lock: lock is kind of a stack. `open` increases the stack. `close` decreases
            the stack. file_object is only closed if the stack is at 0. Ensures consecutive
            method calls works. Would be a problem if the file is closed after each
            sub-operation.
        _mode: file mode
        _driver: file driver
        _comm: MPI communicator
    """

    def __init__(
        self, file_name: str, _comm: mpi.MPI.Comm = mpi.MPI.COMM_WORLD
    ) -> None:
        self.file_object: h5py.File = None
        self.file_name = file_name
        self.simulation_uid = os.path.basename(file_name)
        self._lock = 0
        self._mode = "r"
        self._driver = None
        self._comm = _comm

    def __call__(self, mode: str = "r", driver=None, comm=None) -> FileHandler:
        """Used to set the options for file opening.
        Example: `with sim._file('a', driver='mpio') as file:`
        """
        self._mode = mode
        self._driver = driver
        self._comm = comm if comm is not None else self._comm
        return self

    @capture_key_error
    def __getitem__(self, key) -> Any:
        return self.file_object[key]

    @capture_key_error
    def __delitem__(self, key) -> None:
        del self.file_object[key]

    @capture_key_error
    def __getattr__(self, __name: str) -> Any:
        try:
            return self.file_object.__getattribute__(__name)
        except AttributeError:
            return self.__getattribute__(__name)

    def __enter__(self):
        self.open(self._mode, self._driver, self._comm)
        return self

    def __exit__(self, *args):
        self.close()

    def open(self, mode: str = "r", driver=None, comm=None):
        if self._lock <= 0:
            log.debug(f"[{id(self)}] Open {self.file_name}")
            self.file_object = open_h5file(self.file_name, mode, driver, comm)

        if FILE_MODE_HIRARCHY[self.file_object.mode] < FILE_MODE_HIRARCHY[mode]:
            self.change_file_mode(mode, driver, comm)

        log.debug(f"[{id(self)}] Lock stack {self._lock}")
        self._lock += 1
        return self.file_object

    def close(self):
        self._lock -= 1
        if self._lock == 0:
            log.debug(f"[{id(self)}] Close {self.file_name}")
            self.file_object.close()
        log.debug(f"[{id(self)}] Lock stack {self._lock}")

    def change_file_mode(self, mode: str, driver=None, comm=None):
        log.info(
            f"Forced closing and reopening to change file mode [{self.file_name}]."
        )
        self.file_object.close()
        self.file_object = open_h5file(self.file_name, mode, driver, comm)
