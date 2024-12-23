# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

from pathlib import Path
import pkgutil
import time
from enum import Enum
from functools import total_ordering, wraps
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import h5py
import numpy as np

from bamboost import BAMBOOST_LOGGER
import bamboost
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
        """Context manager to opens the HDF5 file with the specified mode and
        driver.

        This method attempts to open the file, and if it's locked, it will
        retry until the file becomes available.

        Args:
            mode: The mode to open the file with. Defaults to "r" (read-only).
            driver: The driver to use for file I/O. If "mpio" and MPI is
                active, it will use MPI I/O. Defaults to None.

        Returns:
            HDF5File: The opened HDF5 file object.

        Raises:
            BlockingIOError: If the file is locked (handled internally with retries).
        """
        mode = FileMode(mode)
        self._context_stack += 1
        log.debug(f"[{id(self)}] context stack + ({self._context_stack})")

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
                log.debug(f"[{id(self)}] opened file {self._filename} with mode {mode}")
                return self
            except BlockingIOError:
                # If the file is locked, we wait and try again
                log.warning(f"file locked (waiting) --> {self._filename}")
                time.sleep(0.1)

    def close(self):
        self._context_stack -= 1
        if self._context_stack <= 0:
            log.debug(f"[{id(self)}] closing file {self._filename}")
            super().close()
        log.debug(f"[{id(self)}] context stack - ({self._context_stack})")

    def force_close(self):
        super().close()
        self._context_stack = 0

    @property
    def root(self) -> Group:
        return Group("/", self)


_T = TypeVar("_T", bound=h5py.Group | h5py.Dataset | h5py.Datatype)


class H5Reference(Generic[_T]):
    _file: HDF5File

    def __init__(self, name: str, file: HDF5File):
        self._file = file
        self._name = name

    @property
    def _obj(self) -> _T:
        return cast(_T, self._file[self._name])

    @with_file_open(FileMode.READ)
    def __getitem__(
        self, name: str | tuple | slice
    ) -> H5Reference[h5py.Group | h5py.Dataset]:
        return self.new(f"{self._name}/{name}".replace("//", "/"), self._file)

    def open(
        self,
        mode: FileMode | Literal["r", "r+", "w", "w-", "x", "a"] = "r",
        driver: Optional[Literal["mpio"]] = None,
    ) -> HDF5File:
        """Convenience context manager to open the file of this object (see
        HDF5File.open).
        """
        return self._file.open(mode, driver)

    @classmethod
    def new(cls, name: str, file: HDF5File) -> H5Reference:
        """Returns a new pointer object."""
        with file.open(FileMode.READ):
            obj = file[name]
            if isinstance(obj, h5py.Group):
                return Group(name, file)
            elif isinstance(obj, h5py.Dataset):
                return Dataset(name, file)
            else:
                return H5Reference(name, file)

    @property
    @with_file_open()
    def attrs(self):
        return dict(self._obj.attrs)

    @property
    @with_file_open()
    def parent(self) -> Group:
        return Group(self._obj.parent.name, self._file)


class Group(H5Reference[h5py.Group]):
    def __repr__(self) -> str:
        return f'<HDF5 group "{self._name}" (file {self._file._filename})>'

    @property
    def _obj(self) -> h5py.Group:
        obj_group = self._file[self._name]
        assert isinstance(
            obj_group, h5py.Group
        ), f"{self._name} exists but is not a group"
        return obj_group

    @with_file_open(FileMode.READ)
    def keys(self) -> list[str]:
        return list(self._obj.keys())

    def __iter__(self):
        for key in self.keys():
            yield self.__getitem__(key)

    @with_file_open(FileMode.READ)
    def groups(self) -> list[str]:
        return [key for key in self.keys() if isinstance(self._obj[key], h5py.Group)]

    @with_file_open(FileMode.READ)
    def datasets(self) -> list[str]:
        return [key for key in self.keys() if isinstance(self._obj[key], h5py.Dataset)]

    @with_file_open(FileMode.READ)
    def _repr_html_(self):
        """Repr showing the content of the group."""
        html_string = pkgutil.get_data(
            bamboost.__name__, "_repr/hdf5_group.html"
        ).decode()  # type: ignore
        icon = pkgutil.get_data(bamboost.__name__, "_repr/icon.txt").decode()  # type: ignore

        attrs_html = ""
        for key, val in self.attrs.items():
            attrs_html += f"""
            <tr>
                <td>{key}</td>
                <td>{val}</td>
            </tr>
            """
        groups_html = ""
        for key in self.groups():
            groups_html += f"""
            <tr>
                <td>{key}</td>
                <td>{len(self._obj[key])}</td>
            </tr>
            """
        ds_html = ""
        for key in self.datasets():
            obj = self._obj[key]
            ds_html += f"""
            <tr>
                <td>{key}</td>
                <td>{obj.dtype}</td>
                <td>{obj.shape}</td>
            </tr>
            """
        path = self._name
        if not path.startswith("/"):
            path = f"/{path}"
        return (
            html_string.replace("$NAME", path)
            .replace("$UID", Path(self._file._filename).parent.name)
            .replace("$ICON", icon)
            .replace("$attrs", attrs_html)
            .replace("$groups", groups_html)
            .replace("$datasets", ds_html)
            .replace("$version", bamboost.__version__)
        )


class Dataset(H5Reference[h5py.Dataset]):
    def __getitem__(self, key: tuple | slice) -> np.ndarray:
        return h5py.Dataset.__getitem__(self._obj, key)
