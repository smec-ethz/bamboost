"""
This module provides `AttrsDict`, a dictionary-like object that is synced with the
attributes of a group in the HDF5 file.

Key features:
- The object is tied to a simulation. If the simulation is read-only, the object is
  read-only too.
- Automatic synchronization: Updates to the mapping are pushed to the file immediately
  when the simulation is mutable.
- Thread safe and MPI-compatible. Uses the `single_process_queue` of the file object to
  handle attribute updates.
"""

from __future__ import annotations

from typing import Any, Generic, Mapping

import h5py

from bamboost._typing import _MT, Mutable
from bamboost.core.hdf5.file import (
    FileMode,
    HDF5File,
    HDF5Path,
    mutable_only,
    with_file_open,
)


class AttrsDict(Mapping, Generic[_MT]):
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
    _file: HDF5File[_MT]
    _path: str
    _dict: dict

    def __new__(cls, *args, **kwargs):
        if cls is not AttrsDict:
            return super().__new__(cls)

        # Singleton pattern for base attrs dict class
        # signature: __new__(cls, file: HDF5File[_MT], path: str)
        file = args[0] if args else kwargs.get("file")
        path = HDF5Path(args[1]) if len(args) > 1 else kwargs.get("path")

        instances = file._attrs_dict_instances
        if path not in instances:
            instances[path] = super().__new__(cls)
        return instances[path]

    def __init__(self, file: HDF5File[_MT], path: str):
        self._file = file
        self._path = path
        self._dict = self.read()
        self.mutable = file.mutable

    @with_file_open(FileMode.READ)
    def read(self) -> dict:
        return dict(self._obj.attrs)

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

    def __str__(self) -> str:
        return f"<AttrsDict(path={self._path})>"

    def _repr_pretty_(self, p, cycle):
        cls_name = type(self).__name__
        if cycle:
            p.text(f"{cls_name}(...)")
        else:
            with p.group(8, f"{cls_name}(", ")"):
                p.pretty(self._dict)

    @property
    def _obj(self) -> h5py.HLObject:
        obj = self._file[self._path]
        return obj

    @mutable_only
    def __setitem__(self: AttrsDict[Mutable], key: str, value: Any) -> None:
        self._dict[key] = value

        self._file.single_process_queue.add(
            lambda self: self._obj.attrs.__setitem__(key, value),
            (self,),
        )

    @mutable_only
    def __delitem__(self: AttrsDict[Mutable], key: str) -> None:
        del self._dict[key]

        self._file.single_process_queue.add(
            lambda self: self._obj.attrs.__delitem__(key),
            (self,),
        )

    @mutable_only
    def update(self: AttrsDict[Mutable], update_dict: dict) -> None:
        """Update the dictionary. This method pushes the update to the HDF5
        file.

        Args:
            update_dict: new dictionary
        """
        self._dict.update(update_dict)

        self._file.single_process_queue.add(
            lambda self: self._obj.attrs.update(update_dict),
            (self,),
        )
