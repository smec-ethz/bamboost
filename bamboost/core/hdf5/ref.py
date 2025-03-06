"""
This module provides a high-level abstraction for working with HDF5 (`h5py`) groups and
datasets. It is built on the concept of describing an object in the file with a reference
to it (deterministic reference: a file instance, and a path inside the HDF file).

The reference handles file opening and closing, and provides a simple interface to
data, attributes and subgroups. In essence, this model provides `h5py` objects like any
other in-memory data structure.

Classes:
    H5Reference:
        Base class for Groups and Datasets.
    Group:
        Read-only group reference.
    MutableGroup:
        Mutable group reference.
    Dataset:
        Read-only dataset reference.
"""

from __future__ import annotations

import pkgutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from pickletools import string1
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import h5py
import numpy as np
from typing_extensions import Self

import bamboost
from bamboost import BAMBOOST_LOGGER
from bamboost._typing import _MT, Mutable
from bamboost.core.hdf5.attrs_dict import AttrsDict
from bamboost.core.hdf5.file import (
    FileMode,
    FilteredFileMap,
    H5Object,
    HDF5File,
    HDF5Path,
    WriteInstruction,
    mutable_only,
    with_file_open,
)
from bamboost.mpi import MPI_ON

log = BAMBOOST_LOGGER.getChild("hdf5")

MPI_ACTIVE = "mpio" in h5py.registered_drivers() and h5py.get_config().mpi and MPI_ON

_RT_group = TypeVar("_RT_group", bound=Union["Group", "Dataset"])


class H5Reference(H5Object[_MT]):
    _valid: bool | None = None

    def __init__(self, path: str, file: HDF5File[_MT]):
        self._file = file
        self._path = HDF5Path(path)

        # if the file is open, we check if the object exists
        # Otherwise, we assume it exists and check when the file is opened
        if file.is_open:
            self._valid = self._path in file

    @property
    def _obj(self) -> Union[h5py.Group, h5py.Dataset, h5py.Datatype]:
        _obj = self._file[self._path]
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
        return f'<HDF5 {type(self).__name__} "{self._path}" ({valid_str}, file {self._file._filename})>'

    @overload
    def __getitem__(self, value: str) -> Union[Self, Dataset]: ...
    @overload
    def __getitem__(self, value: Union[slice, Tuple[()]]) -> Any: ...
    @overload
    def __getitem__(self, value: tuple[str, Type[Group]]) -> Group[_MT]: ...
    @overload
    def __getitem__(self, value: tuple[str, Type[Dataset]]) -> Dataset[_MT]: ...
    @with_file_open(FileMode.READ)
    def __getitem__(self, value):
        obj = self._obj
        assert not isinstance(obj, h5py.Datatype), (
            "__getitem__ not implemented for Datatype"
        )

        # If the value is a slice or empty tuple, we return the sliced dataset
        if isinstance(value, slice) or (isinstance(value, tuple) and len(value) == 0):
            return obj[value]

        # Here we know that we are looking for a group or dataset
        if isinstance(value, tuple):
            name, _type = value
        else:
            name, _type = cast(str, value), None
        return self.new(self._path / name, self._file, _type)

    def open(self, *args, **kwargs) -> HDF5File[_MT]:
        """Convenience context manager to open the file of this object (see
        HDF5File.open).
        """
        return self._file.open(*args, **kwargs)

    @classmethod
    def new(
        cls,
        path: str,
        file: HDF5File[_MT],
        _type: Optional[Type[_RT_group]] = None,
    ) -> _RT_group:
        """Returns a new pointer object."""
        if _type:
            return _type(path, file)

        with file.open(FileMode.READ):
            _obj = file[path]
            if isinstance(_obj, h5py.Group):
                return cast(_RT_group, Group(path, file))
            elif isinstance(_obj, h5py.Dataset):
                return cast(_RT_group, Dataset(path, file))
            else:
                raise ValueError(f"Object {path} is not a group or dataset")

    @cached_property
    def attrs(self) -> AttrsDict[_MT]:
        return AttrsDict(self._file, self._path)

    @property
    @with_file_open(FileMode.READ)
    def parent(self) -> Group[_MT]:
        return Group(self._obj.parent.name or "", self._file)

    @property
    def mutable(self) -> bool:
        return self._file.mutable


class Group(H5Reference[_MT]):
    def __init__(self, path, file):
        super().__init__(path, file)

        # Create a subset view of the file map with all objects
        self._group_map = FilteredFileMap(file.file_map, path)

    @mutable_only
    def __setitem__(self: Group[Mutable], key, newvalue):
        """Used to set an attribute.
        Will be written as an attribute to the group.
        """
        if isinstance(newvalue, np.ndarray):
            self.add_numerical_dataset(key, np.array(newvalue))
        else:
            self.attrs.__setitem__(key, newvalue)

    @mutable_only
    @with_file_open(FileMode.APPEND)
    def __delitem__(self: Group[Mutable], key) -> None:
        """Deletes an item."""
        if key in self.attrs.keys():
            del self._obj.attrs[key]
        else:
            self._file.delete_object(self._path / key)

    def _ipython_key_completions_(self):
        return self.keys()

    @property
    def _obj(self) -> h5py.Group:
        _obj = super()._obj
        if not isinstance(_obj, h5py.Group):
            raise ValueError(f"Object {self._path} is not a group")
        return _obj

    def __iter__(self):
        for key in self.keys():
            yield self.__getitem__(key)

    def _assert_file_map_is_valid(self):
        if not self._group_map.valid:
            with self._file.open(FileMode.READ):
                self._group_map.file_map.populate()

    def keys(self):
        self._assert_file_map_is_valid()
        return self._group_map.keys()

    def groups(self):
        self._assert_file_map_is_valid()
        return self._group_map.groups()

    def datasets(self):
        self._assert_file_map_is_valid()
        return self._group_map.datasets()

    def items(
        self,
    ) -> Generator[Tuple[str, Union[Group[_MT], Dataset[_MT]]], None, None]:
        for key in self.keys():
            yield key, self.__getitem__(key)

    @with_file_open(FileMode.READ)
    def _repr_html_(self):
        """Repr showing the content of the group."""
        # If the object is not a group, return a simple representation
        try:
            _obj = self._obj
        except ValueError:
            self._valid = False
            return f"Invalid HDF5 object: <b>{self._path}</b> is not a group"
        except KeyError:
            self._valid = False
            return f"Invalid HDF5 object: <b>{self._path}</b> not found in file"

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

    @mutable_only
    @with_file_open(FileMode.APPEND)
    def require_self(self: Group[Mutable]) -> None:
        """Create the group if it doesn't exist yet."""
        self._file.require_group(self._path)
        self._file.file_map[self._path] = h5py.Group

    @overload
    def require_group(
        self: Group[Mutable],
        name: str,
        *,
        return_type: Type[_RT_group],
    ) -> _RT_group: ...
    @overload
    def require_group(self: Group[Mutable], name: str) -> Group[Mutable]: ...
    @mutable_only
    @with_file_open(FileMode.APPEND, driver="mpio")
    def require_group(self, name, *, return_type=None):
        """Create a group if it doesn't exist yet."""
        self._obj.require_group(name)

        # update file_map
        self._group_map[name] = h5py.Group

        return self.new(self._path.joinpath(name), self._file, _type=return_type)

    @mutable_only
    def require_dataset(
        self: Group[Mutable],
        name: str,
        shape: tuple[int, ...],
        dtype,
        exact: bool = False,
        **kwargs,
    ) -> h5py.Dataset:
        grp = self._obj.require_dataset(name, shape, dtype, exact=exact, **kwargs)
        self._group_map[name] = h5py.Dataset  # type: ignore
        return grp

    @mutable_only
    def add_numerical_dataset(
        self: Group[Mutable],
        name: str,
        vector: np.ndarray,
        attrs: Optional[Dict[str, Any]] = None,
        dtype: Optional[str] = None,
        *,
        file_map: bool = True,
    ) -> None:
        """Add a dataset to the group. Error is thrown if attempting to overwrite
        with different shape than before. If same shape, data is overwritten
        (this is inherited from h5py -> require_dataset)

        Args:
            name: Name for the dataset
            vector: Data to write (max 2d)
            attrs: Optional. Attributes of dataset.
            dtype: Optional. dtype of dataset. If not specified, uses dtype of inpyt array
            file_map: Optional. If True, the dataset is added to the file map. Default is True.
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

        with self._file.open(FileMode.APPEND, driver="mpio"):
            dataset = self._obj.require_dataset(
                name, shape=vec_shape, dtype=dtype if dtype else vector.dtype
            )
            dataset[idx_start:idx_end] = vector

            class WriteAttrs(WriteInstruction):
                @staticmethod
                def __call__():
                    self._obj[name].attrs.update(attrs)

            self.post_write_instruction(WriteAttrs())

        log.info(f'Written dataset to "{self._path}/{name}"')

        # update file_map
        if file_map:
            self._group_map[name] = h5py.Dataset

    @mutable_only
    def add_dataset(
        self: Group[Mutable],
        name: str,
        data: Any,
        attrs: Optional[Dict[str, Any]] = None,
        dtype: Optional[str] = None,
    ) -> None:
        if self._file._comm.rank == 0:
            with self._file.open(FileMode.APPEND):
                self._obj.create_dataset(name, data=data, dtype=dtype)
                if attrs:
                    self._obj[name].attrs.update(attrs)
        log.info(f'Written dataset to "{self._path}/{name}"')

        # update file_map
        self._group_map[name] = h5py.Dataset


class MutableGroup(Group[Mutable]):
    @overload
    def __getitem__(self, key: str): ...
    @overload
    def __getitem__(self, key: tuple[str, Type[_RT_group]]) -> _RT_group: ...
    @with_file_open(FileMode.READ)
    def __getitem__(self, key):
        """Used to access datasets (:class:`~bamboost.common.hdf_pointer.Dataset`)
        or groups inside this group (:class:`~bamboost.common.hdf_pointer.MutableGroup`)
        """
        if key in self.attrs:
            return self.attrs[key]

        return super().__getitem__(key)


class Dataset(H5Reference[_MT]):
    @property
    def _obj(self) -> h5py.Dataset:
        obj = super()._obj
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(f"Object {self._path} is not a dataset")
        return obj

    @with_file_open(FileMode.READ)
    def __getitem__(self, key: tuple | slice | int) -> Any:
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
