from __future__ import annotations

from collections.abc import KeysView, Mapping, MutableMapping
from typing import TYPE_CHECKING, Iterator, Type

import h5py

from bamboost.core.hdf5.hdf5path import HDF5Path

if TYPE_CHECKING:
    from bamboost.core.hdf5.file import HDF5File

_VT_filemap = Type[h5py.Group | h5py.Dataset]


class KeysViewHDF5(KeysView):
    def __str__(self) -> str:
        return "<KeysViewHDF5 {}>".format(list(self))

    __repr__ = __str__


class _FileMapMixin(Mapping[HDF5Path, _VT_filemap]):
    def keys(self) -> KeysViewHDF5:
        return KeysViewHDF5(self)

    def datasets(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.items() if v is h5py.Dataset)

    def groups(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.items() if v is h5py.Group)

    def _ipython_key_completions_(self):
        return self.keys()

    def _repr_pretty_(self, p, _cycle):
        p.pretty(dict(self))

    def __repr__(self):
        return f"{type(self).__name__}({dict(self)})"


class FileMap(MutableMapping[HDF5Path, _VT_filemap], _FileMapMixin):
    def __init__(self, file: HDF5File):
        self._file = file

        self._dict: dict[HDF5Path, _VT_filemap] = {HDF5Path("/"): h5py.Group}
        """A cache of all known paths in the file and their types (Group or Dataset).
        Paths are absolute."""

        self._children: dict[HDF5Path, set[str]] = {}
        """A cache of immediate children names for each group path."""

        self._expanded_groups: set[HDF5Path] = set()
        """A set of group paths that have been expanded (i.e., their children have been
        fetched from the file)."""

    def __getitem__(self, key: str, /) -> _VT_filemap:
        path = HDF5Path(key)
        if path not in self._dict:
            # If the item is not in the cache, try expanding its parent
            parent = path.parent
            if parent not in self._expanded_groups:
                self.expand_group(parent)

        return self._dict[HDF5Path(key)]

    def __setitem__(self, key: str, value: _VT_filemap) -> None:
        path = HDF5Path(key)
        self._dict[path] = value
        parent = path.parent
        if parent in self._children:
            self._children[parent].add(path.name)

    def __delitem__(self, key: str) -> None:
        path = HDF5Path(key)
        self._dict.pop(path, None)
        parent = path.parent
        if parent in self._children:
            self._children[parent].discard(path.name)
        # Also remove if it was a group with cached children
        self._children.pop(path, None)
        self._expanded_groups.discard(path)

    def __iter__(self) -> Iterator[HDF5Path]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def expand_group(self, path: HDF5Path) -> None:
        """Fetch immediate children of the given path from the HDF5 file and cache them."""
        if path in self._expanded_groups:
            return

        with self._file.open():
            try:
                obj = self._file[str(path)]
                self._dict[path] = type(obj)

                if not isinstance(obj, h5py.Group):
                    self._expanded_groups.add(path)
                    return

                if path not in self._children:
                    self._children[path] = set()

                for name in obj.keys():
                    child_path = path.joinpath(name)
                    # Use get(name, getclass=True) to get the type without opening the object
                    child_type = obj.get(name, getclass=True)
                    self._dict[child_path] = child_type
                    self._children[path].add(name)

                self._expanded_groups.add(path)
            except KeyError:
                # Path doesn't exist in file
                pass

    def populate(self, *, exclude_numeric: bool = False) -> None:
        """Eagerly visit all groups and datasets to cache them.
        Assumes the file is open.
        """
        self._dict[HDF5Path("/")] = h5py.Group

        def cache_items(name, _obj):
            path = HDF5Path(name)
            if exclude_numeric and path.basename.isdigit():
                return
            self._dict[path] = type(_obj)
            parent = path.parent
            if parent not in self._children:
                self._children[parent] = set()
            self._children[parent].add(path.name)

        self._file.visititems(cache_items)

    def invalidate(self) -> None:
        self._dict.clear()
        self._dict[HDF5Path("/")] = h5py.Group
        self._children.clear()
        self._expanded_groups.clear()


class FilteredFileMap(MutableMapping[HDF5Path, _VT_filemap], _FileMapMixin):
    def __init__(self, file_map: FileMap, parent: str) -> None:
        self.parent = HDF5Path(parent)
        self.file_map = file_map

    def __getitem__(self, key: str, /):
        return self.file_map[self.parent.joinpath(key)]

    def __setitem__(self, key: str, value):
        self.file_map[self.parent.joinpath(key)] = value

    def __delitem__(self, key):
        del self.file_map[self.parent.joinpath(key)]

    def children(self) -> Iterator[HDF5Path]:
        return self.__iter__()

    def children_groups(self) -> Iterator[HDF5Path]:
        return filter(lambda path: self[path] is h5py.Group, self)

    def children_datasets(self) -> Iterator[HDF5Path]:
        return filter(lambda path: self[path] is h5py.Dataset, self)

    def __iter__(self):
        if self.parent not in self.file_map._expanded_groups:
            self.file_map.expand_group(self.parent)

        child_names = self.file_map._children.get(self.parent, set())
        for name in sorted(child_names):
            yield HDF5Path(name, absolute=False)

    def __len__(self):
        if self.parent not in self.file_map._expanded_groups:
            self.file_map.expand_group(self.parent)

        return len(self.file_map._children.get(self.parent, set()))
