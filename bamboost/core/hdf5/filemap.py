from __future__ import annotations

from collections.abc import ItemsView, KeysView, Mapping, MutableMapping
from typing import TYPE_CHECKING, Callable, Generator, Iterator, Type, Union

import h5py

from bamboost.core.hdf5.hdf5path import HDF5Path

if TYPE_CHECKING:
    from bamboost.core.hdf5.file import HDF5File

_VT_filemap = Type[Union[h5py.Group, h5py.Dataset]]


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

    def populate(self, *, exclude_numeric: bool = False) -> None:
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


class FilteredFileMap(MutableMapping[HDF5Path, _VT_filemap], _FileMapMixin):
    def __init__(self, file_map: FileMap, parent: str) -> None:
        self.parent = HDF5Path(parent)
        self.file_map = file_map

    @property
    def valid(self) -> bool:
        return self.file_map.valid

    @valid.setter
    def valid(self, value: bool) -> None:
        self.file_map.valid = value

    def __getitem__(self, key: str, /):
        return self.file_map[self.parent.joinpath(key)]

    def __setitem__(self, key: str, value):
        self.file_map[self.parent.joinpath(key)] = value

    def __delitem__(self, key):
        del self.file_map[self.parent.joinpath(key)]

    def children(self) -> Iterator[HDF5Path]:
        return filter(lambda path: "/" not in path, self)

    def children_groups(self) -> Iterator[HDF5Path]:
        return filter(lambda path: self[path] is h5py.Group, self.children())

    def children_datasets(self) -> Iterator[HDF5Path]:
        return filter(lambda path: self[path] is h5py.Dataset, self.children())

    def __iter__(self):
        return map(
            lambda x: HDF5Path(x).relative_to(self.parent),
            filter(
                lambda x: HDF5Path(x).is_child_of(self.parent),
                self.file_map,
            ),
        )

    def __len__(self):
        return sum(1 for _ in self.__iter__())
