from __future__ import annotations

from typing import Any, Generic, Mapping, cast

import h5py

from bamboost._typing import _MT, Mutable
from bamboost.core.hdf5.file import (
    FileMode,
    HDF5File,
    mutable_only,
    with_file_open,
)


class GroupDict(Mapping, Generic[_MT]):
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

    def __init__(self, file: HDF5File[_MT], path: str) -> None:
        self._file = file
        self._path = path
        self._dict = self.read()
        self.mutable = self._file.mutable

    @with_file_open(FileMode.READ)
    def read(self) -> dict:
        tmp_dict = dict()

        try:
            grp = cast(h5py.Group, self._file[self._path])
        except KeyError:
            raise KeyError(
                f"Group {self._path} not found in file {self._file._filename}."
            )

        # Read in attributes
        tmp_dict.update(grp.attrs)

        # Read in datasets
        for key, value in grp.items():
            if not isinstance(value, h5py.Dataset):
                continue
            tmp_dict.update({key: value[()]})

        return tmp_dict

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def _ipython_key_completions_(self):
        return tuple(self._dict.keys())

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

    @property
    def _obj(self) -> h5py.Group:
        obj = self._file[self._path]
        assert isinstance(obj, h5py.Group), f"Object at {self._path} is not a group."
        return obj

    @mutable_only
    def __setitem__(self: GroupDict[Mutable], key: str, value: Any) -> None:
        self._dict[key] = value

        with self._file.open(FileMode.APPEND, root_only=True):
            self._obj.attrs[key] = value

    @mutable_only
    def __delitem__(self: GroupDict[Mutable], key: str) -> None:
        with self._file.open(FileMode.APPEND, root_only=True):
            del self._obj.attrs[key]

    @mutable_only
    def update(self: GroupDict[Mutable], update_dict: dict) -> None:
        """Update the dictionary. This method pushes the update to the HDF5
        file.

        Args:
            update_dict: new dictionary
        """
        self._dict.update(update_dict)

        with self._file.open(FileMode.APPEND, root_only=True):
            self._obj.attrs.update(update_dict)
