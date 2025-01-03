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
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Optional,
    Protocol,
    Type,
    Union,
    overload,
)

import h5py
from typing_extensions import Self

from bamboost import BAMBOOST_LOGGER
from bamboost._typing import _MT, Immutable, Mutable
from bamboost.mpi import MPI, MPI_ON
from bamboost.utilities import StrPath

if TYPE_CHECKING:
    from bamboost.mpi import Comm

    from .ref import Group

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


class HDF5Path(str):
    def __new__(cls, path: str, absolute: bool = True):
        if isinstance(path, HDF5Path):
            return path
        prefix = "/" if absolute else ""
        return super().__new__(cls, prefix + "/".join(filter(None, path.split("/"))))

    def __truediv__(self, other: str) -> HDF5Path:
        return self.joinpath(other)

    def joinpath(self, other: str) -> HDF5Path:
        return HDF5Path(f"{self}/{other}")

    def relative_to(self, other: HDF5Path) -> HDF5Path:
        return HDF5Path(self[len(other) :], absolute=False)

    @property
    def parent(self) -> str:
        return self.rsplit("/", 1)[0] or "/"

    @property
    def path(self) -> PurePosixPath:
        return PurePosixPath(self)


class HasFileAttribute(Protocol):
    _file: HDF5File


def mutable_only(func):
    """Decorator to raise an error if the file is not mutable."""

    @wraps(func)
    def wrapper(self: HasFileAttribute, *args, **kwargs):
        if not self._file._mutable:
            raise PermissionError("Simulation file is read-only.")
        return func(self, *args, **kwargs)

    wrapper._mutable_only = True  # type: ignore
    return wrapper


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


class HDF5File(h5py.File, Generic[_MT]):
    """Wrapper for h5py.File to add some functionality"""

    _filename: str
    _comm: Comm
    _context_stack: int = 0
    _object_map: dict[HDF5Path, Type[h5py.Group | h5py.Dataset]]
    _mapped: bool = False
    _mutable: bool
    _current_root_only_flag: bool = False

    @overload
    def __init__(
        self: HDF5File[Immutable],
        file: StrPath,
        comm: Optional[Comm] = None,
        mutable: Literal[False] = False,
    ): ...
    @overload
    def __init__(
        self: HDF5File[Mutable],
        file: StrPath,
        comm: Optional[Comm] = None,
        mutable: Literal[True] = True,
    ): ...
    def __init__(
        self,
        file: StrPath,
        comm: Optional[Comm] = None,
        mutable: bool = False,
    ):
        self._filename = file.as_posix() if isinstance(file, Path) else file
        self._comm = comm or MPI.COMM_WORLD
        self._object_map = dict()
        self._mutable = mutable

    def __repr__(self) -> str:
        mode_info = self.mode if self.is_open else "proxy"
        status = "open" if self.is_open else "closed"
        mutability = Mutable if self._mutable else Immutable
        return (
            f'<{mutability} HDF5 file "{self._filename}" (mode {mode_info}, {status})>'
        )

    @overload
    def open(
        self: HDF5File[Immutable],
        mode: Literal[FileMode.READ, "r"] = "r",
        driver: Optional[Literal["mpio"]] = None,
        *,
        root_only: bool = False,
    ) -> HDF5File[Immutable]: ...
    @overload
    def open(
        self: HDF5File[Mutable],
        mode: Union[FileMode, Literal["r", "r+", "w", "w-", "x", "a"]] = "r",
        driver: Optional[Literal["mpio"]] = None,
        *,
        root_only: bool = False,
    ) -> HDF5File[Mutable]: ...
    @overload
    def open(
        self: HDF5File,
        mode: Union[FileMode, str] = "r",
        driver: Optional[Literal["mpio"]] = None,
        *,
        root_only: bool = False,
    ) -> HDF5File: ...
    def open(
        self,
        mode: Union[FileMode, str] = "r",
        driver=None,
        *,
        root_only=False,
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

        assert not (
            root_only and driver == "mpio"
        ), "Cannot use driver 'mpio' and root_only together."

        if not self._mutable and mode > FileMode.READ:
            log.error(
                f"File is read-only, cannot open in mode {mode.value} (open in read-only mode)"
            )
            mode = FileMode.READ

        self._context_stack += 1
        log.debug(f"[{id(self)}] context stack + ({self._context_stack})")

        if self.is_open:
            if mode > FileMode(self.mode):
                # if the new operation requires a higher mode, we close the file
                # to reopen it with the new mode
                super().close()
            else:
                # if the file is already open with the same or higher mode, we
                # just increase the context stack and return
                return self
        else:
            # Used by the context manager of this class
            # the root_only flag is ignored if the file is already open
            self._current_root_only_flag = root_only

        if self._current_root_only_flag and self._comm.rank != 0:  # do not open
            return self

        return self._try_open_repeat(mode, driver)

    def __enter__(self):
        if self._current_root_only_flag and self._comm.rank != 0:
            raise StopIteration()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is StopIteration and self._comm.rank != 0:
            return
        return super().__exit__(exc_type, exc_value, traceback)

    def _try_open_repeat(
        self, mode: FileMode, driver: Optional[Literal["mpio"]] = None
    ) -> Self:
        # try to open the file until it is available
        while True:
            try:
                if driver == "mpio" and MPI_ACTIVE:
                    h5py.File.__init__(
                        self, self._filename, mode.value, driver=driver, comm=self._comm
                    )
                else:
                    h5py.File.__init__(self, self._filename, mode.value)
                log.debug(
                    f"[{id(self)}] opened file (mode {mode.value}) {self._filename}"
                )

                # cache object names
                if not self._mapped:
                    self._map_objects()
                    self._mapped = True

                return self
            except BlockingIOError:
                # If the file is locked, we wait and try again
                log.warning(f"file locked (waiting) --> {self._filename}")
                time.sleep(0.1)

    def map_objects(self) -> Self:
        with self.open(FileMode.READ):
            self._map_objects()
        return self

    def _map_objects(self) -> None:
        objects = dict()

        # visit all groups and datasets to cache them
        def cache_items(name, _obj):
            objects[HDF5Path(name)] = type(_obj)

        self.visititems(cache_items)
        self._object_map = objects

    def close(self):
        self._context_stack = max(0, self._context_stack - 1)
        log.debug(f"[{id(self)}] context stack - ({self._context_stack})")

        # if the context stack is 0, we close the file
        if self._context_stack <= 0:
            log.debug(f"[{id(self)}] closed file {self._filename}")
            super().close()

    def force_close(self):
        super().close()
        self._context_stack = 0

    @property
    def is_open(self) -> bool:
        return bool(hasattr(self, "id") and self.id.valid)

    @property
    def root(self) -> Group[_MT]:
        from .ref import Group

        return Group("/", self)
