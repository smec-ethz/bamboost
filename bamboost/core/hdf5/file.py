"""
This module provides an enhanced interface for working with HDF5 files. Its main goal is
to provide automatic file management with heavy caching to limit file access to a minimum.

Key features:
- Lazy h5py.File wrapper: `HDF5File` postpones file opening until necessary.
- Cached file map: `FileMap` caches all groups and datasets in the file. We also implement
  a singleton pattern for the file map to avoid multiple instances of the same file. This
  allows to subsequent usage of the file map (and the file in general) while skipping file
  access.
- Automatic handling of opening the file when necessary using a context stack, which
  enables knowing when the file is no longer needed.
- Queued operations enable the bundling of operations. This is useful to bundle operations
  which require to be executed on the root process only (such as attribute updating). The
  queued operations are executed once the serial context is closed. If the file is not
  used when something is added to the queue, it is executed immediately.
- Utility classes for working with HDF5 paths.
- Decorators to handle file opening, and file mutability checks (avoid write when not
  intended).

Attributes:
    MPI_ACTIVE (bool): Indicates whether MPI support is available for HDF5.
    log (Logger): Logger instance for this module.

Classes:
    FileMode:
        Enum representing different file modes (READ, WRITE, APPEND, etc.).
    HDF5Path:
        A specialized string class to handle HDF5-style paths.
    HasFile (Protocol):
        Protocol defining objects that have an associated HDF5 file.
    KeysViewHDF5:
        A specialized `KeysView` implementation for HDF5 mappings.
    FileMap:
        Manages an in-memory cache of groups and datasets within an HDF5 file.
    FilteredFileMap:
        A filtered view of `FileMap`, allowing scoped access within a subpath.
    ProcessQueue:
        A queue to defer execution of file operations, especially useful in MPI contexts.
    HDF5File:
        A wrapper around `h5py.File` with additional functionality, including:
        - Context-managed opening/closing
        - Deferred process execution
        - Mutability enforcement
        - Custom file path handling

Decorators:
    mutable_only:
        Ensures that a method can only be executed if the file is mutable.
    with_file_open:
        Opens and closes the file automatically when executing a method.
    add_to_file_queue:
        Adds a method call to the process queue instead of executing immediately.
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from collections import deque
from contextlib import contextmanager
from enum import Enum
from functools import total_ordering, wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    overload,
)

import h5py
from typing_extensions import Concatenate, Self

from bamboost import BAMBOOST_LOGGER, config
from bamboost._typing import _MT, _P, _T, Immutable, Mutable
from bamboost.core.hdf5.filemap import FileMap
from bamboost.core.hdf5.hdf5path import HDF5Path
from bamboost.mpi import MPI, MPI_ON, Communicator
from bamboost.mpi.utilities import RootProcessMeta
from bamboost.plugins import ElligibleForPlugin
from bamboost.utilities import StrPath

if TYPE_CHECKING:
    from bamboost.core.hdf5.attrsdict import AttrsDict
    from bamboost.core.hdf5.ref import Group
    from bamboost.mpi import Comm

    class HasFile(Protocol[_MT]):
        _file: HDF5File[_MT]

    _T_HasFile = TypeVar("_T_HasFile", bound=HasFile)


log = BAMBOOST_LOGGER.getChild("hdf5")
"""Logger instance for this module."""

HDF_MPI_ACTIVE = (
    "mpio" in h5py.registered_drivers() and h5py.get_config().mpi and MPI_ON
)
"""Indicates whether MPI support is available for HDF5."""


@total_ordering
class FileMode(Enum):
    READ = "r"
    READ_WRITE = "r+"
    APPEND = "a"
    WRITE = "w"
    WRITE_FAIL = "w-"
    WRITE_CREATE = "x"

    __hirarchy__ = {"r": 0, "r+": 1, "a": 1, "w": 1, "w-": 1, "x": 1}

    def __lt__(self, other) -> bool:
        return self.__hirarchy__[self.value] < self.__hirarchy__[other.value]

    def __eq__(self, other) -> bool:
        return self.__hirarchy__[self.value] == self.__hirarchy__[other.value]


def mutable_only(
    method: Callable[Concatenate[_T_HasFile, _P], _T],
) -> Callable[Concatenate[_T_HasFile, _P], _T]:
    """Decorator to raise an error if the file is not mutable."""

    @wraps(method)
    def inner(self: _T_HasFile, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if not self._file.mutable:
            raise PermissionError("Simulation file is read-only.")
        return method(self, *args, **kwargs)

    inner._mutable_only = True  # type: ignore
    return inner


def with_file_open(
    mode: FileMode = FileMode.READ,
    driver: Optional[Literal["mpio"]] = None,
) -> Callable[
    [Callable[Concatenate[_T_HasFile, _P], _T]],
    Callable[Concatenate[_T_HasFile, _P], _T],
]:
    """Decorator for context manager to open and close the file for a method of a class
    with a file attribute (self._file).

    Args:
        mode: The mode to open the file with. Defaults to FileMode.READ.
        driver: The driver to use for file I/O. If "mpio" and MPI is active, it will
            use MPI I/O. Defaults to None.
    """

    def decorator(
        method: Callable[Concatenate[_T_HasFile, _P], _T],
    ) -> Callable[Concatenate[_T_HasFile, _P], _T]:
        @wraps(method)
        def inner(self: _T_HasFile, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            with self._file.open(mode, driver):
                return method(self, *args, **kwargs)

        return inner

    return decorator


def add_to_file_queue(
    method: Callable[Concatenate[_T_H5Object, _P], None],
) -> Callable[Concatenate[_T_H5Object, _P], None]:
    """Decorator to add a method call to the single process queue of the file object
    instead of executing it immediately.
    """

    @wraps(method)
    def inner(self: _T_H5Object, *args: _P.args, **kwargs: _P.kwargs) -> None:
        self.post_write_instruction(lambda: method(self, *args, **kwargs))

    return inner


class H5Object(Generic[_MT], ElligibleForPlugin):
    _file: HDF5File[_MT]
    _comm = Communicator()

    def __init__(self, file: HDF5File[_MT]) -> None:
        self._file = file

    @overload
    def mutable(self: H5Object[Mutable]) -> Literal[True]: ...
    @overload
    def mutable(self: H5Object[Immutable]) -> Literal[False]: ...
    @property
    def mutable(self) -> bool:
        return self._file.mutable

    def open(
        self,
        mode: FileMode | str = "r",
        driver: Optional[Literal["mpio"]] = None,
    ) -> HDF5File[_MT]:
        """Use this as a context manager in a `with` statement.
        Purpose: keeping the file open to directly access/edit something in the
        HDF5 file of this simulation.

        Args:
            mode: file mode (see h5py docs)
            driver: file driver (see h5py docs)
        """
        return self._file.open(mode, driver=driver)

    @mutable_only
    def post_write_instruction(self, instruction: Callable[[], None]) -> None:
        if self._file.available_for_single_process_write():
            # call instruction immediately
            return self._file.single_process_queue.apply_instruction(instruction)

        self._file.single_process_queue.add_instruction(instruction)

    @contextmanager
    def suspend_immediate_write(self) -> Generator[None, None, None]:
        """Context manager to suspend immediate write operations. Patches
        self._file.available_for_single_process_write to return False."""
        original_method = self._file.available_for_single_process_write
        self._file.available_for_single_process_write = lambda: False
        try:
            yield
        finally:
            self._file.available_for_single_process_write = original_method
            self._file.single_process_queue.apply()


_T_H5Object = TypeVar("_T_H5Object", bound=H5Object)


class SingleProcessQueue(deque[Callable[[], None]], metaclass=RootProcessMeta):
    """A queue to defer execution of write operations that need to be executed on the root
    only. Only relevant for parallelized code.

    This class is a deque of instructions that are to be executed in order when the file is
    available for writing (i.e., not open with MPI I/O OR closed). We append instructions
    to the right and pop them from the left.

    This class uses the RootProcessMeta metaclass to ensure that all methods are only
    executed on the root process.
    """

    _comm = Communicator()

    def __init__(self, file: HDF5File):
        self._file = file
        super().__init__()

    def add_instruction(self, instruction: Callable[[], None]) -> None:
        self.append(instruction)
        log.debug(f"Added {type(instruction).__name__} to process queue")

    @with_file_open(FileMode.APPEND)
    def apply_instruction(self, instruction: Callable[[], None]) -> None:
        log.debug(f"Applying {type(instruction).__name__}")
        instruction()

    def apply(self) -> None:
        """Applies all write instructions in the queue."""
        if not self:
            log.debug("SingleProcessQueue is empty")
            return

        log.debug("Applying process queue")
        with self._file.open(FileMode.APPEND):
            while self:
                instruction = self.popleft()
                log.debug(f"Applying {type(instruction).__name__}")
                instruction()


class WriteInstruction(ABC):
    """Abstract base class for write instructions. Not useful currently, but could be
    extended in the future (e.g. provide logging)."""

    def __init__(self): ...
    def __call__(self) -> None: ...


class HDF5File(h5py.File, Generic[_MT]):
    """Lazy `h5py.File` wrapper with deferred process execution and file map caching.

    Args:
        file: The path to the HDF5 file.
        comm: The MPI communicator. Defaults to MPI.COMM_WORLD.
        mutable: Whether the file is mutable. Defaults to False.
    """

    _filename: str
    _comm = Communicator()
    _context_stack: int = 0
    mutable: bool
    file_map: FileMap
    _is_open_on_root_only: bool = False
    _attrs_dict_instances: dict[str, "AttrsDict[_MT]"] = {}

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
        self._path = Path(self._filename).absolute()
        self.file_map = FileMap(self)
        self.mutable = mutable

        # if the file is immutable, we immediately check if it exists.
        # if it doesn't, we raise an exception
        if not mutable and not self._path.exists():
            raise FileNotFoundError(f"File {self._filename} does not exist.")

    def __repr__(self) -> str:
        mode_info = self.mode if self.is_open else "proxy"
        status = "open" if self.is_open else "closed"
        mutability = Mutable if self.mutable else Immutable
        return (
            f'<{mutability} HDF5 file "{self._filename}" (mode {mode_info}, {status})>'
        )

    def _create_file(self: HDF5File[Mutable]) -> HDF5File[Mutable]:
        """Opens and closes the file to create it while doing nothing to it."""
        with self.open("a"):
            pass
        return self

    @overload
    def open(
        self: HDF5File[Immutable],
        mode: Literal[FileMode.READ, "r"] = "r",
        driver: Optional[Literal["mpio"]] = None,
    ) -> HDF5File[Immutable]: ...
    @overload
    def open(
        self: HDF5File[Mutable],
        mode: Union[FileMode, Literal["r", "r+", "w", "w-", "x", "a"]] = "r",
        driver: Optional[Literal["mpio"]] = None,
    ) -> HDF5File[Mutable]: ...
    @overload
    def open(
        self: HDF5File,
        mode: Union[FileMode, str] = "r",
        driver: Optional[Literal["mpio"]] = None,
    ) -> HDF5File: ...
    def open(
        self,
        mode: Union[FileMode, str] = "r",
        driver=None,
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
            FileNotFoundError: If the file does not exist and you're trying to open it read-only.
        """
        mode = FileMode(mode)

        if (mode > FileMode.READ) and (not self.mutable):
            log.error(f"File is read-only, cannot open with mode {mode.value}")
            raise PermissionError(
                "Attempted to open read-only file with illegal file mode."
            )

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

        return self._try_open_repeat(mode, driver)

    def _try_open_repeat(
        self, mode: FileMode, driver: Optional[Literal["mpio"]] = None
    ) -> Self:
        waiting_logged = False

        # try to open the file until it is available
        while True:
            try:
                if HDF_MPI_ACTIVE and mode > FileMode.READ and driver == "mpio":
                    h5py.File.__init__(
                        self, self._filename, mode.value, driver=driver, comm=self._comm
                    )
                else:
                    h5py.File.__init__(self, self._filename, mode.value)
                log.debug(
                    f"[{id(self)}] opened file (mode {mode.value}) {self._filename}"
                )

                # create file map
                if not self.file_map.valid:
                    self.file_map.populate()

                return self
            except BlockingIOError:
                # If the file is locked, we wait and try again
                if not waiting_logged:
                    level = logging._nameToLevel[config.options.log_file_lock_severity]
                    log.log(
                        level, f"[{id(self)}] file locked (waiting) {self._filename}"
                    )
                waiting_logged = True
                time.sleep(0.01)

    def close(self):
        self._context_stack = max(0, self._context_stack - 1)

        # if the context stack is 0, we close the file
        if self._context_stack <= 0:
            log.debug(f"[{id(self)}] closed file {self._filename}")
            super().close()

            # if the file is mutable, we further apply the single process queue
            if self.mutable:
                self.single_process_queue.apply()

        log.debug(f"[{id(self)}] context stack - ({self._context_stack})")

    def force_close(self):
        super().close()
        self._context_stack = 0

    def delete_object(self, path: Union[HDF5Path, str]) -> None:
        """Deletes an object in the file. In addition to deleting the object, revoking
        this method also removes the object from the file map.

        Args:
            path: The path to the object to delete.
        """
        # call h5py's delete method
        super().__delitem__(str(path))
        # remove from file map
        del self.file_map[str(path)]

    @property
    def is_open(self) -> bool:
        return bool(hasattr(self, "id") and self.id.valid)

    @property
    def single_process_queue(self) -> SingleProcessQueue:
        """The single process queue of this file object. See `SingleProcessQueue` for
        details.
        """
        try:
            return self._single_process_queue
        except AttributeError:
            self._single_process_queue = SingleProcessQueue(self)
            return self._single_process_queue

    def available_for_single_process_write(self) -> bool:
        """Whether single process write instructions can be executed immediately."""
        return (not self.is_open) or (self.driver != "mpio")

    @property
    def root(self) -> Group[_MT]:
        """Returns the root group of the file. Same as `Group("/", file)`"""
        from .ref import Group

        return Group("/", self)
