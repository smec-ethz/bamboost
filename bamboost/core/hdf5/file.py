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

import time
from collections import deque
from collections.abc import ItemsView, KeysView, Mapping, MutableMapping
from enum import Enum
from functools import total_ordering, wraps
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    overload,
)

import h5py
from typing_extensions import Concatenate, Self

from bamboost import BAMBOOST_LOGGER
from bamboost._typing import _MT, _P, _T, Immutable, Mutable
from bamboost.mpi import MPI, MPI_ON
from bamboost.mpi.utilities import RootProcessMeta
from bamboost.utilities import StrPath

if TYPE_CHECKING:
    from bamboost.mpi import Comm

    from .attrs_dict import AttrsDict
    from .ref import Group

log = BAMBOOST_LOGGER.getChild("hdf5")

MPI_ACTIVE = "mpio" in h5py.registered_drivers() and h5py.get_config().mpi and MPI_ON


_VT_filemap = Type[Union[h5py.Group, h5py.Dataset]]


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


class HDF5Path(str):
    def __new__(cls, path: str, absolute: bool = True):
        if isinstance(path, HDF5Path):
            return path
        prefix = "/" if absolute else ""
        return super().__new__(cls, prefix + "/".join(filter(None, path.split("/"))))

    def __truediv__(self, other: str) -> HDF5Path:
        return self.joinpath(other)

    def joinpath(self, *other: str) -> HDF5Path:
        return HDF5Path("/".join([self, *other]))

    def relative_to(self, other: Union[HDF5Path, str]) -> HDF5Path:
        other = HDF5Path(other)
        if not self.startswith(other):
            raise ValueError(f"{self} is not a subpath of {other}")
        return HDF5Path(self[len(other) :], absolute=False)

    @property
    def parent(self) -> HDF5Path:
        return HDF5Path(self.rsplit("/", 1)[0] or "/")

    @property
    def basename(self) -> str:
        return self.rsplit("/", 1)[-1]

    @property
    def path(self) -> PurePosixPath:
        return PurePosixPath(self)


class HasFile(Protocol[_MT]):
    _file: HDF5File[_MT]


HasFileSubtype = TypeVar("HasFileSubtype", bound=HasFile)


def mutable_only(
    method: Callable[Concatenate[HasFileSubtype, _P], _T],
) -> Callable[Concatenate[HasFileSubtype, _P], _T]:
    """Decorator to raise an error if the file is not mutable."""

    @wraps(method)
    def inner(self: HasFileSubtype, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if not self._file.mutable:
            raise PermissionError("Simulation file is read-only.")
        return method(self, *args, **kwargs)

    inner._mutable_only = True  # type: ignore
    return inner


def with_file_open(
    mode: FileMode = FileMode.READ,
    driver: Optional[Literal["mpio"]] = None,
) -> Callable[
    [Callable[Concatenate[HasFileSubtype, _P], _T]],
    Callable[Concatenate[HasFileSubtype, _P], _T],
]:
    """Decorator for context manager to open and close the file for a method of a class
    with a file attribute (self._file).

    Args:
        mode: The mode to open the file with. Defaults to FileMode.READ.
        driver: The driver to use for file I/O. If "mpio" and MPI is active, it will
            use MPI I/O. Defaults to None.
    """

    def decorator(
        method: Callable[Concatenate[HasFileSubtype, _P], _T],
    ) -> Callable[Concatenate[HasFileSubtype, _P], _T]:
        @wraps(method)
        def inner(self: HasFileSubtype, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            with self._file.open(mode, driver):
                return method(self, *args, **kwargs)

        return inner

    return decorator


def add_to_file_queue(
    method: Callable[Concatenate[HasFileSubtype, _P], None],
) -> Callable[Concatenate[HasFileSubtype, _P], None]:
    @wraps(method)
    def inner(self: HasFileSubtype, *args: _P.args, **kwargs: _P.kwargs) -> None:
        self._file.single_process_queue.add(method, (self, *args, *kwargs))

    return inner


class KeysViewHDF5(KeysView):
    def __str__(self) -> str:
        return "<KeysViewHDF5 {}>".format(list(self))

    __repr__ = __str__


class _FileMapMixin(Mapping[str, _VT_filemap]):
    def keys(self) -> KeysViewHDF5:
        return KeysViewHDF5(self)

    def items(self, all: bool = False) -> ItemsView[str, _VT_filemap]:
        if not all:
            return ItemsView({k: v for k, v in super().items() if "/" not in k})
        return super().items()

    def datasets(self):
        return tuple(k for k, v in self.items() if v is h5py.Dataset)

    def groups(self):
        return tuple(k for k, v in self.items() if v is h5py.Group)

    def _ipython_key_completions_(self):
        return self.keys()


class FileMap(MutableMapping[str, _VT_filemap], _FileMapMixin):
    _valid: bool

    def __init__(self, file: HDF5File):
        self._file = file
        self._dict: dict[HDF5Path, _VT_filemap] = {}
        self.valid = False

    def __getitem__(self, key: str, /) -> _VT_filemap:
        return self._dict[HDF5Path(key)]

    def __setitem__(self, key: str, value: _VT_filemap) -> None:
        self._dict[HDF5Path(key)] = value

    def __delitem__(self, key: str) -> None:
        self._dict.pop(HDF5Path(key))

    def __iter__(self) -> Iterator[HDF5Path]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def populate(self, *, exclude_numeric: bool = True) -> None:
        """Assumes the file is open."""

        # visit all groups and datasets to cache them
        def cache_items(name, _obj):
            path = HDF5Path(name)
            if exclude_numeric and path.basename.isdigit():
                return
            self._dict[path] = type(_obj)

        self._file.visititems(cache_items)
        self.valid = True

    def invalidate(self) -> None:
        self.valid = False

    def items(self) -> ItemsView[str, _VT_filemap]:
        return super().items(all=True)


class FilteredFileMap(MutableMapping[str, _VT_filemap], _FileMapMixin):
    def __init__(self, file_map: FileMap, parent: str) -> None:
        self.parent = HDF5Path(parent)
        self.file_map = file_map
        self.valid = self.file_map.valid

    def __getitem__(self, key, /):
        return self.file_map[self.parent.joinpath(key)]

    def __setitem__(self, key: str, value):
        self.file_map[self.parent.joinpath(key)] = value

    def __delitem__(self, key):
        del self.file_map[self.parent.joinpath(key)]

    def __iter__(self):
        return map(
            lambda x: HDF5Path(x).relative_to(self.parent),
            filter(
                lambda x: x.startswith(self.parent) and not x == self.parent,
                self.file_map,
            ),
        )

    def __len__(self):
        return sum(1 for _ in self.__iter__())


class ProcessQueue(metaclass=RootProcessMeta):
    def __init__(self, file: HDF5File):
        self._file = file
        self._comm = file._comm  # needed for MPISafeMeta to work
        self.deque = deque[tuple[Callable, tuple]]()

    @RootProcessMeta.exclude
    def add(self, func: Callable, args: tuple) -> None:
        # if file is not open, open it and apply the function immediately
        if not self._file.is_open:
            if self._file._comm.rank == 0:
                with self._file.open(FileMode.APPEND):
                    func(*args)
                    return
        # if file is open and not using mpio, apply the function immediately
        elif self._file.driver != "mpio":
            func(*args)
            return
        # else, the file is open with mpio, so we add the function to the queue
        self.deque.append((func, args))
        log.debug(
            f"Added {func.__qualname__} to process queue (args: {','.join(map(str, args))})"
        )

    def apply(self):
        if not self.deque:
            log.debug("Process queue is empty")
            return

        with self._file.open(FileMode.APPEND):
            log.debug("Applying process queue...")
            while self.deque:
                func, args = self.deque.popleft()
                func(*args)
                log.debug(
                    f"Applied {func.__qualname__} (args: {','.join(map(str, args))})"
                )


class HDF5File(h5py.File, Generic[_MT]):
    """Wrapper for h5py.File to add some functionality"""

    _filename: str
    _comm: Comm
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
        self._comm = comm or MPI.COMM_WORLD
        self._path = Path(self._filename).absolute()
        self.file_map = FileMap(self)
        self.mutable = mutable

        # Single process queue: Stuff in here is applied when the file is closed (at
        # latest)
        if self.mutable:
            self.single_process_queue = ProcessQueue(self)

    def __repr__(self) -> str:
        mode_info = self.mode if self.is_open else "proxy"
        status = "open" if self.is_open else "closed"
        mutability = Mutable if self.mutable else Immutable
        return (
            f'<{mutability} HDF5 file "{self._filename}" (mode {mode_info}, {status})>'
        )

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
        """
        mode = FileMode(mode)

        if not self.mutable and mode > FileMode.READ:
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

        return self._try_open_repeat(mode, driver)

    def _try_open_repeat(
        self, mode: FileMode, driver: Optional[Literal["mpio"]] = None
    ) -> Self:
        waiting_logged = False

        # try to open the file until it is available
        while True:
            try:
                if MPI_ACTIVE and mode > FileMode.READ and driver == "mpio":
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
                    log.warning(f"[{id(self)}] file locked (waiting) {self._filename}")
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
    def root(self) -> Group[_MT]:
        from .ref import Group

        return Group("/", self)
