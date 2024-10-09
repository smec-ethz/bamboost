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
from collections.abc import Iterable
from typing import Any, Literal

import h5py
import numpy as np

import bamboost
from bamboost.common.file_handler import FileHandler, with_file_open

__all__ = [
    "BasePointer",
    "Group",
    "MutableGroup",
    "Dataset",
]

log = bamboost.BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


class BasePointer:
    """Pointer to a location in an hdf5 file. The constructor takes a
    :class:`~.file_handler.FileHandler` and the in-file path to the object.
    The base class represents a generic group in the file

    Args:
        file_handler: file this belongs to
        path_to_data: infile path to object
    """

    @classmethod
    def new_pointer(cls, file_handler: FileHandler, path_to_data: str) -> BasePointer:
        """Returns a new pointer object."""
        with file_handler("r") as f:
            obj = f.file_object[path_to_data]
            if isinstance(obj, h5py.Group):
                if issubclass(cls, Group):
                    return cls(file_handler, path_to_data)
                else:
                    return Group(file_handler, path_to_data)
            elif isinstance(obj, h5py.Dataset):
                return Dataset(file_handler, path_to_data)
            else:
                return BasePointer(file_handler, path_to_data)

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        self._file = file_handler
        self.path_to_data = path_to_data

    @property
    def obj(self):
        """The object this BasePointer points to. File needs to be open for access."""
        return self._file[self.path_to_data]

    @with_file_open("r")
    def __str__(self) -> str:
        try:
            return (
                f"{self.__class__} pointing to: "
                + self._file[self.path_to_data].__repr__()
            )
        except KeyError:
            return f"{self.__class__} pointing to: {self.path_to_data} (No data at that location!)"

    __repr__ = __str__

    @with_file_open("r")
    def __getattr__(self, __name: str) -> Any:
        """Any attribute request is sent to the h5py object the pointer points to."""
        if hasattr(self.obj, __name):
            return self.obj.__getattribute__(__name)
        return self.__getattribute__(__name)

    @with_file_open("r")
    def __getitem__(self, key):
        new_path = f"{self.path_to_data}/{key}"
        return self.new_pointer(self._file, new_path)

    @property
    @with_file_open()
    def attrs(self):
        return dict(self.obj.attrs)


class Group(BasePointer):
    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)

    def _ipython_key_completions_(self):
        return self.keys()

    def __iter__(self):
        for key in self.keys():
            yield self.__getitem__(key)

    @with_file_open("r")
    def keys(self) -> set:
        return set(self.obj.keys())

    @with_file_open("r")
    def groups(self) -> set:
        return {key for key in self.keys() if isinstance(self.obj[key], h5py.Group)}

    @with_file_open("r")
    def datasets(self) -> set:
        return {key for key in self.keys() if isinstance(self.obj[key], h5py.Dataset)}

    @with_file_open("r")
    def extract_attrs(
        self, variant: Literal["group", "dataset", "all"] = "all"
    ) -> dict:
        """Extract the attributes of all members of the group.

        Args:
            variant: one of 'group', 'dataset', 'all'
        """
        if variant == "group":
            keys = self.groups()
        elif variant == "dataset":
            keys = self.datasets()
        elif variant == "all":
            keys = self.keys()
        else:
            raise ValueError("variant must be one of 'group', 'dataset', 'all'")

        attrs = {}
        for key in keys:
            attrs[key] = dict(self.obj[key].attrs)
        return attrs

    @with_file_open("r")
    def _repr_html_(self):
        """Repr showing the content of the group."""
        html_string = pkgutil.get_data(
            bamboost.__name__, "html/hdf5_group.html"
        ).decode()
        icon = pkgutil.get_data(bamboost.__name__, "html/icon.txt").decode()

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
                <td>{len(self.obj[key])}</td>
            </tr>
            """
        ds_html = ""
        for key in self.datasets():
            obj = self.obj[key]
            ds_html += f"""
            <tr>
                <td>{key}</td>
                <td>{obj.dtype}</td>
                <td>{obj.shape}</td>
            </tr>
            """
        path = self.path_to_data
        if not path.startswith("/"):
            path = f"/{path}"
        return (
            html_string.replace("$NAME", path)
            .replace("$UID", self._file.simulation_uid)
            .replace("$ICON", icon)
            .replace("$attrs", attrs_html)
            .replace("$groups", groups_html)
            .replace("$datasets", ds_html)
            .replace("$version", bamboost.__version__)
        )


class MutableGroup(Group):
    """Used for the `userdata` group."""

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)
        # Create group if it doesn't exist
        with self._file("a", driver="mpio"):
            self._file.file_object.require_group(path_to_data)

    def _ipython_key_completions_(self):
        return self.keys().union(set(self.attrs.keys()))

    @with_file_open("r")
    def __getitem__(self, key) -> Any:
        """Used to access datasets (:class:`~bamboost.common.hdf_pointer.Dataset`)
        or groups inside this group (:class:`~bamboost.common.hdf_pointer.MutableGroup`)
        """
        try:
            return super().__getitem__(key)
        except KeyError:
            pass

        try:
            return self.obj.attrs[key]
        except KeyError:
            pass

        return super().__getitem__(key)

    def __setitem__(self, key, newvalue):
        """Used to set an attribute.
        Will be written as an attribute to the group.
        """
        if isinstance(newvalue, str) or not isinstance(newvalue, Iterable):
            self.update_attrs({key: newvalue})
        else:
            self.add_dataset(key, np.array(newvalue))

    @with_file_open("a")
    def __delitem__(self, key) -> None:
        """Deletes an item."""
        if key in self.attrs.keys():
            del self.obj.attrs[key]
        else:
            del self.obj[key]

    @with_file_open("a")
    def update_attrs(self, attrs: dict) -> None:
        """Update the attributes of the group.

        Args:
            attrs: the dictionary to write as attributes
        """
        self.obj.attrs.update(attrs)

    def add_dataset(
        self, name: str, vector: np.ndarray, attrs: dict = None, dtype: str = None
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

        with self._file("a", driver="mpio"):
            dataset = self.obj.require_dataset(
                name, shape=vec_shape, dtype=dtype if dtype else vector.dtype
            )
            dataset[idx_start:idx_end] = vector
            for key, item in attrs.items():
                dataset.attrs[key] = item

        log.info(f"Written {name} as userdata to {self._file.file_name}...")

    def require_group(self, name: str) -> MutableGroup:
        """Add a new group to the current group. If exists, return existing.

        Returns:
            :class:`~bamboost.hdf_pointer.Group`
        """
        return MutableGroup(self._file, f"{self.path_to_data}/{name}")


class Dataset(BasePointer):
    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)

    @with_file_open("r")
    def __getitem__(self, slice):
        return self.obj[slice]

    @property
    @with_file_open()
    def attrs(self):
        return dict(self.obj.attrs)

    @property
    @with_file_open()
    def shape(self):
        return self.obj.shape

    @property
    @with_file_open()
    def dtype(self):
        return self.obj.dtype
