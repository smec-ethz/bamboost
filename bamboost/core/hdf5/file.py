# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import pkgutil
import time
from enum import Enum
from functools import cache, cached_property, total_ordering, wraps
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import h5py
import numpy as np

import bamboost
from bamboost import BAMBOOST_LOGGER
from bamboost.mpi import MPI, MPI_ON
from bamboost.utilities import StrPath

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


class HDF5Path(str):
    def __new__(cls, path: str, absolute: bool = True):
        if isinstance(path, HDF5Path):
            return path
        prefix = "/" if absolute else ""
        return super().__new__(cls, prefix + "/".join(filter(None, path.split("/"))))

    def __truediv__(self, other: str) -> HDF5Path:
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
    _object_map: dict[HDF5Path, Type[h5py.Group | h5py.Dataset]]
    _readonly: bool = True
    _mapped: bool = False

    def __init__(
        self,
        file: StrPath,
        comm: Optional[Comm] = None,
        *,
        readonly: bool = True,
    ):
        self._filename = file.as_posix() if isinstance(file, Path) else file
        self._comm = comm or MPI.COMM_WORLD
        self._object_map = dict()
        self._readonly = readonly

    def __repr__(self) -> str:
        mode_info = self.mode if self.is_open else "proxy"
        status = "open" if self.is_open else "closed"
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
        if self._readonly and mode > FileMode.READ:
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

        # try to open the file until it is available
        while True:
            try:
                if driver == "mpio" and MPI_ACTIVE:
                    super().__init__(
                        self._filename, mode.value, driver=driver, comm=self._comm
                    )
                else:
                    super().__init__(self._filename, mode.value)
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

    def map_objects(self) -> HDF5File:
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
    def root(self) -> Group:
        return Group("/", self)


_T = TypeVar("_T", bound=Union[h5py.Group, h5py.Dataset])
_R = TypeVar("_R", bound=Union["Group", "Dataset"])


class H5Reference(Generic[_T]):
    _file: HDF5File
    _type: str = "object"
    _valid: bool | None = None

    def __init__(self, path: str, file: HDF5File):
        self._file = file
        self._path = HDF5Path(path)

        # if the file is open, we check if the object exists
        if file.is_open:
            self._valid = self._path in file

    @property
    def _obj(self) -> _T:
        _obj = cast(_T, self._file[self._path])
        self._valid = True
        return _obj

    def __repr__(self) -> str:
        valid_str = (
            "not checked"
            if self._valid is None
            else "valid"
            if self._valid
            else "invalid"
        )
        return f'<HDF5 {self._type} "{self._path}" ({valid_str}, file {self._file._filename})>'

    @overload
    def __getitem__(self, value: str) -> Union[Group, Dataset]: ...
    @overload
    def __getitem__(self, value: tuple | slice): ...
    @overload
    def __getitem__(self, value: tuple[str, Type[_R]]) -> _R: ...
    @with_file_open(FileMode.READ)
    def __getitem__(self, value: Union[str, tuple[str, Type[_R]], tuple | slice]) -> _R:
        if isinstance(value, tuple):
            name, _type = value
        else:
            name, _type = cast(str, value), None
        return self.new(self._path / name, self._file, _type)

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
    def new(cls, name: str, file: HDF5File, _type: Optional[Type[_R]] = None) -> _R:
        """Returns a new pointer object."""
        with file.open(FileMode.READ):
            _obj = file[name]
            if isinstance(_obj, h5py.Group):
                return cast(_R, Group(name, file))
            elif isinstance(_obj, h5py.Dataset):
                return cast(_R, Dataset(name, file))
            else:
                raise ValueError(f"Object {name} is not a group or dataset")

    @cached_property
    @with_file_open(FileMode.READ)
    def attrs(self) -> dict[str, Any]:
        return dict(self._obj.attrs)

    @property
    @with_file_open()
    def parent(self) -> Group:
        return Group(self._obj.parent.name or "", self._file)


class Group(H5Reference[h5py.Group]):
    _type: str = "group"

    def __init__(self, path: str, file: HDF5File):
        super().__init__(path, file)

    def _ipython_key_completions_(self):
        return self.keys(all=True)

    @property
    def _obj(self) -> h5py.Group:
        _obj = super()._obj
        if not isinstance(_obj, h5py.Group):
            raise ValueError(f"Object {self._path} is not a group")
        return _obj

    def __iter__(self):
        for key in self.keys():
            yield self.__getitem__(key)

    @cached_property
    def _object_map(self) -> dict[HDF5Path, Type[h5py.Group | h5py.Dataset]]:
        _ = self._file._mapped or self._file.map_objects()
        return {
            k: v
            for k, v in self._file._object_map.items()
            if k.startswith(self._path) and k != self._path
        }

    @property
    def _children_map(self) -> dict[HDF5Path, Type[h5py.Group | h5py.Dataset]]:
        return {k: v for k, v in self._object_map.items() if k.parent == self._path}

    @cache
    def keys(self, all: bool = False) -> list[HDF5Path]:
        object_map = self._object_map if all else self._children_map
        return [path.relative_to(self._path) for path in object_map.keys()]

    @cache
    def groups(self, all: bool = False) -> list[HDF5Path]:
        object_map = self._object_map if all else self._children_map
        return [
            name.relative_to(self._path)
            for name, t in object_map.items()
            if t == h5py.Group
        ]

    @cache
    def datasets(self, all: bool = False) -> list[str]:
        object_map = self._object_map if all else self._children_map
        return [
            name.relative_to(self._path)
            for name, t in object_map.items()
            if t == h5py.Dataset
        ]

    @with_file_open(FileMode.READ)
    def _repr_html_(self):
        """Repr showing the content of the group."""
        # If the object is not a group, return a simple representation
        try:
            _obj = self._obj
        except ValueError:
            self._valid = False
            return f"Invalid HDF5 object: <b>{self._path}</b> is not a group"

        from jinja2 import Template

        attrs = dict(_obj.attrs)
        groups = {key: len(_obj[self._path / key]) for key in self.groups()}  # type: ignore
        datasets = {
            key: (_obj[self._path / key].dtype, _obj[self._path / key].shape)  # type: ignore
            for key in self.datasets()
        }

        path = self._path
        path = path if path[0] == "/" else "/" + path

        html_template = pkgutil.get_data(
            bamboost.__name__, "_repr/hdf5_group.html"
        ).decode()  # type: ignore
        icon = pkgutil.get_data(bamboost.__name__, "_repr/icon.txt").decode()  # type: ignore
        template = Template(html_template)

        return template.render(
            uid=Path(self._file._filename).parent.name,
            name=path,
            icon=icon,
            version=bamboost.__version__,
            attrs=attrs,
            groups=groups,
            datasets=datasets,
        )


class MutableGroup(Group):
    def __init__(self, name: str, file: HDF5File):
        if file._readonly:
            raise ValueError("Cannot modify read-only file")

        super().__init__(name, file)

    @with_file_open(FileMode.READ)
    def __getitem__(self, key: str) -> Group | Dataset:
        """Used to access datasets (:class:`~bamboost.common.hdf_pointer.Dataset`)
        or groups inside this group (:class:`~bamboost.common.hdf_pointer.MutableGroup`)
        """
        if key in self.keys():
            return self.new(f"{self._path}/{key}", self._file)

        if key in self.attrs:
            return self.attrs[key]

        return super().__getitem__(key)

    def __setitem__(self, key, newvalue):
        """Used to set an attribute.
        Will be written as an attribute to the group.
        """
        if isinstance(newvalue, str) or not isinstance(newvalue, Iterable):
            self.update_attrs({key: newvalue})
        else:
            self.add_dataset(key, np.array(newvalue))

    @with_file_open(FileMode.APPEND)
    def __delitem__(self, key) -> None:
        """Deletes an item."""
        if key in self.attrs.keys():
            del self._obj.attrs[key]
        else:
            del self._obj[key]

    @with_file_open(FileMode.APPEND)
    def update_attrs(self, attrs: Dict[str, Any]) -> None:
        """Update the attributes of the group.

        Args:
            attrs: the dictionary to write as attributes
        """
        self._obj.attrs.update(attrs)

    @with_file_open(FileMode.APPEND)
    def require_group(self, name: str) -> MutableGroup:
        """Create a group if it doesn't exist yet."""
        self._obj.require_group(name)
        return self.new(name, self._file)

    def add_dataset(
        self,
        name: str,
        vector: np.ndarray,
        attrs: Optional[Dict[str, Any]] = None,
        dtype: Optional[str] = None,
    ) -> None:
        """Add a dataset to the group. Error is thrown if attempting to overwrite
        with different shape than before. If same shape, data is overwritten
        (this is inherited from h5py -> require_dataset)

        Args:
            name: Name for the dataset
            vector: Data to write (max 2d)
            attrs: Optional. Attributes of dataset.
            dtype: Optional. dtype of dataset. If not specified, uses dtype of inpyt array
        """
        if attrs is None:
            attrs = {}
        length_local = vector.shape[0]
        length_p = np.array(self._file._comm.allgather(length_local))
        length = np.sum(length_p)
        dim = vector.shape[1:]
        vec_shape = length, *dim

        ranks = np.array([i for i in range(self._file._comm.size)])
        idx_start = np.sum(length_p[ranks < self._file._comm.rank])
        idx_end = idx_start + length_local

        # with self._file("a", driver="mpio"):
        with self.open(FileMode.APPEND, driver="mpio"):
            dataset = self._obj.require_dataset(
                name, shape=vec_shape, dtype=dtype if dtype else vector.dtype
            )
            dataset[idx_start:idx_end] = vector

        self.update_attrs(attrs)
        log.info(f'Written dataset to "{self._path}/{name}"')


class Dataset(H5Reference[h5py.Dataset]):
    _type: str = "dataset"

    @with_file_open(FileMode.READ)
    def __getitem__(self, key: tuple | slice) -> np.ndarray:
        return h5py.Dataset.__getitem__(self._obj, key)

    @property
    @with_file_open(FileMode.READ)
    def shape(self):
        return self._obj.shape

    @property
    @with_file_open(FileMode.READ)
    def dtype(self):
        return self._obj.dtype


# Aliases for Group and Dataset
_g = Group
_d = Dataset
