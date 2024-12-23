# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import time
from enum import Enum
from functools import total_ordering, wraps
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Protocol,
    Union,
)

import h5py

from bamboost import BAMBOOST_LOGGER
from bamboost.mpi import MPI, MPI_ON

if TYPE_CHECKING:
    from bamboost.mpi import Comm

log = BAMBOOST_LOGGER.getChild("hdf5")

MPI_ACTIVE = "mpio" in h5py.registered_drivers() and h5py.get_config().mpi and MPI_ON


@total_ordering
class FileMode(Enum):
    READ = "r"
    READ_WRITE = "r+"
    APPEND = "a"
    WRITE = "w"
    WRITE_FAIL = "w-"
    WRITE_CREATE = "x"

    __hirarchy__ = {"r": 0, "r+": 1, "a": 1, "w": 2, "w-": 3, "x": 3}

    def __lt__(self, other) -> bool:
        return self.__hirarchy__[self.value] < self.__hirarchy__[other.value]

    def __eq__(self, other) -> bool:
        return self.__hirarchy__[self.value] == self.__hirarchy__[other.value]


class HasFileAttribute(Protocol):
    _file: HDF5File


def with_file_open(
    mode: FileMode = FileMode.READ,
    driver: Optional[Literal["mpio"]] = None,
):
    """Decorator for context manager to open and close the file for a method of
    a class with a file attribute (self._file)
    """

    def decorator(method):
        @wraps(method)
        def inner(self: HasFileAttribute, *args, **kwargs):
            with self._file.open(mode, driver):
                return method(self, *args, **kwargs)

        return inner

    return decorator


class HDF5File(h5py.File):
    """Wrapper for h5py.File to add some functionality"""

    _filename: str
    _comm: Comm
    _context_stack: int = 0

    def __init__(
        self,
        file: str,
        comm: Optional[Comm] = None,
    ):
        self._filename = file
        self._comm = comm or MPI.COMM_WORLD

    def __repr__(self) -> str:
        mode_info = self.mode if hasattr(self, "id") and self.id.valid else "proxy"
        status = "open" if hasattr(self, "id") and self.id.valid else "closed"
        return f'<HDF5 file "{self._filename}" (mode {mode_info}, {status})>'

    def open(
        self,
        mode: Union[FileMode, Literal["r", "r+", "w", "w-", "x", "a"]] = "r",
        driver: Optional[Literal["mpio"]] = None,
    ) -> HDF5File:
        mode = FileMode(mode)
        self._context_stack += 1
        log.debug(f"context stack increased to {self._context_stack}")

        if hasattr(self, "id") and self.id.valid:
            if mode > FileMode(self.mode):
                # if the new operation requires a higher mode, we close the file
                # to reopen it with the new mode
                super().close()
            else:
                # if the file is already open with the same or higher mode, we
                # just increase the context stack and return
                return self

        # try to open the file until it is available
        while True:
            try:
                if driver == "mpio" and MPI_ACTIVE:
                    super().__init__(
                        self._filename, mode.value, driver=driver, comm=self._comm
                    )
                else:
                    super().__init__(self._filename, mode.value)
                return self
            except BlockingIOError:
                # If the file is locked, we wait and try again
                log.warning(f"file locked (waiting) --> {self._filename}")
                time.sleep(0.1)

    def close(self):
        self._context_stack -= 1
        if self._context_stack <= 0:
            log.debug(f"closing file {self._filename}")
            super().close()
        log.debug(f"context stack decreased to {self._context_stack}")

    def force_close(self):
        super().close()
        self._context_stack = 0
