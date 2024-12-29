from __future__ import annotations

from functools import wraps
from typing import Any, Mapping, cast

import h5py

from bamboost.core import utilities
from bamboost.core.hdf5.file import FileMode, HDF5File, with_file_open


def _mutable_only(func):
    """Decorator to raise an error if the object is not mutable."""

    @wraps(func)
    def wrapper(self: GroupDict, *args, **kwargs):
        if self._file._mutable:
            raise PermissionError("Simulation is read-only.")
        return func(self, *args, **kwargs)

    return wrapper


class GroupDict(Mapping):
    """A dictionary-like object for the attributes of a group in the HDF5
    file.

    This object is tied to a simulation. If the simulation is read-only, the
    object is immutable. If mutable, changes are pushed to the HDF5 file
    immediately.

    Args:
        simulation: the simulation object
        path: path to the group in the HDF5 file
    """

    mutable: bool = False

    def __init__(self, file: HDF5File, path: str) -> None:
        self._file = file
        self._path = path
        self._dict = self.read()
        self.mutable = not self._file._mutable

    @with_file_open(FileMode.READ)
    def read(self) -> dict:
        tmp_dict = dict()

        try:
            grp = cast(h5py.Group, self._file[self._path])
        except KeyError:
            raise KeyError(
                f"Group {self._path} not found in file {self._file._filename}."
            )

        tmp_dict.update(grp.attrs)
        for key, value in grp.items():
            if not isinstance(value, h5py.Dataset):
                continue
            tmp_dict.update({key: value[()]})

        return utilities.unflatten_dict(tmp_dict)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self) -> str:
        return self._dict.__repr__()

    def _repr_pretty_(self, p, cycle):
        cls_name = type(self).__name__
        if cycle:
            p.text(f"{cls_name}(...)")
        else:
            with p.group(8, f"{cls_name}(", ")"):
                p.pretty(self._dict)

    def _ipython_key_completions_(self):
        return tuple(self._dict.keys())

    @property
    def _obj(self) -> h5py.Group:
        obj = self._file[self._path]
        assert isinstance(obj, h5py.Group), f"Object at {self._path} is not a group."
        return obj

    # Methods for mutable objects only
    @_mutable_only
    def __setitem__(self, key: str, value: Any) -> None:
        self._dict[key] = value
        with self._file.open(FileMode.APPEND):
            self._obj.attrs[key] = value

    @_mutable_only
    def __delitem__(self, key: str) -> None:
        with self._file.open(FileMode.APPEND):
            del self._obj.attrs[key]
