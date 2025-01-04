from __future__ import annotations

import pkgutil
from functools import cached_property, wraps
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
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
from bamboost.core.hdf5.dict import AttrsDict
from bamboost.core.hdf5.file import (
    FileMode,
    FilteredFileMap,
    HDF5File,
    HDF5Path,
    mutable_only,
    with_file_open,
)
from bamboost.mpi import MPI_ON

log = BAMBOOST_LOGGER.getChild("hdf5")

MPI_ACTIVE = "mpio" in h5py.registered_drivers() and h5py.get_config().mpi and MPI_ON

_RT_group = TypeVar("_RT_group", bound=Union["Group", "Dataset"])


class H5Reference(Generic[_MT]):
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
    def __getitem__(self, value: Union[slice, Tuple[()]]) -> np.ndarray: ...
    @overload
    def __getitem__(self, value: tuple[str, Type[Group]]) -> Self: ...
    @overload
    def __getitem__(self, value: tuple[str, Type[Dataset]]) -> Dataset[_MT]: ...
    @with_file_open(FileMode.READ)
    def __getitem__(self, value):
        obj = self._obj
        assert not isinstance(
            obj, h5py.Datatype
        ), "__getitem__ not implemented for Datatype"

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
        cls, path: str, file: HDF5File[_MT], _type: Optional[Type[_RT_group]] = None
    ) -> _RT_group:
        """Returns a new pointer object."""
        with file.open(FileMode.READ):
            _obj = file[path]
            if isinstance(_obj, h5py.Group):
                return cast(_RT_group, cls(path, file))
            elif isinstance(_obj, h5py.Dataset):
                return cast(_RT_group, Dataset(path, file))
            else:
                raise ValueError(f"Object {path} is not a group or dataset")

    @cached_property
    def attrs(self) -> AttrsDict[_MT]:
        return AttrsDict(self._file, self._path)

    @property
    @with_file_open(FileMode.READ)
    def parent(self) -> Self:
        return self.__class__(self._obj.parent.name or "", self._file)

    @property
    def mutable(self) -> bool:
        return self._file.mutable


class Group(H5Reference[_MT]):
    def __init__(self, path, file):
        super().__init__(path, file)

        # Create a subset view of the file map with all objects
        self._group_map = FilteredFileMap(file, path)

    @mutable_only
    def __setitem__(self: Group[Mutable], key, newvalue):
        """Used to set an attribute.
        Will be written as an attribute to the group.
        """
        if isinstance(newvalue, str) or not isinstance(newvalue, Iterable):
            self.update_attrs({key: newvalue})
        else:
            self.add_numerical_dataset(key, np.array(newvalue))

    @mutable_only
    @with_file_open(FileMode.APPEND)
    def __delitem__(self: Group[Mutable], key) -> None:
        """Deletes an item."""
        if key in self.attrs.keys():
            del self._obj.attrs[key]
        else:
            del self._obj[key]

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

    @mutable_only
    @with_file_open(FileMode.APPEND)
    def update_attrs(self: Group[Mutable], attrs: Dict[str, Any]) -> None:
        """Update the attributes of the group.

        Args:
            attrs: the dictionary to write as attributes
        """
        self._obj.attrs.update(attrs)

    @mutable_only
    @with_file_open(FileMode.APPEND)
    def require_group(self: Group[Mutable], name: str) -> Group[Mutable]:
        """Create a group if it doesn't exist yet."""
        self._obj.require_group(name)

        # update file_map
        self._group_map[name] = h5py.Group

        return self.new(self._path.joinpath(name), self._file)

    @mutable_only
    def add_numerical_dataset(
        self: Group[Mutable],
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

        # update file_map
        self._group_map[name] = h5py.Dataset

    @mutable_only
    def add_dataset(
        self: Group[Mutable],
        name: str,
        data: Any,
        attrs: Optional[Dict[str, Any]] = None,
        dtype: Optional[str] = None,
    ) -> None:
        with self._file.open(FileMode.APPEND, root_only=True):
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
