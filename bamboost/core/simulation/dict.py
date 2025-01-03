from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, Generator, cast

import numpy as np

from bamboost.core import utilities
from bamboost.core.hdf5.dict import GroupDict, mutable_only
from bamboost.core.hdf5.file import (
    _MT,
    FileMode,
    HDF5File,
    Mutable,
)
from bamboost.index import _SimulationMetadataT

if TYPE_CHECKING:
    from bamboost.core.simulation.base import _Simulation


class Parameters(GroupDict[_MT]):
    _simulation: _Simulation

    def __init__(self, simulation: _Simulation[_MT]) -> None:
        file = simulation._file
        super().__init__(file, "/parameters")
        self._simulation = simulation

    def read(self) -> dict:
        tmp_dict = super().read()
        self._ipython_keys_ = tuple(tmp_dict.keys())
        return utilities.unflatten_dict(tmp_dict)

    def _ipython_key_completions_(self) -> Generator[str, None, None]:
        for key, obj in self._dict.items():
            yield key
            if isinstance(obj, dict):
                for subkey in obj.keys():
                    yield f"{key}.{subkey}"

    def __getitem__(self, key: str) -> Any:
        """Get a parameter. Can use nested access with dot notation."""
        return reduce(lambda obj, k: obj[k], key.split("."), self._dict)

    @mutable_only
    def __setitem__(self: Parameters[Mutable], key: str, value: Any) -> None:
        active_dict = reduce(lambda obj, k: obj[k], key.split(".")[:-1], self._dict)
        active_dict[key.split(".")[-1]] = value

        with self._file.open(FileMode.APPEND, root_only=True):
            # because values can be stored as datasets or attributes, we need
            # to check if the key already exists and remove it before writing
            # the new value -> to avoid duplicates
            try:
                del self._obj.attrs[key]  # remove existing attribute
            except KeyError:
                pass
            try:
                del self._obj[key]  # remove existing dataset
            except KeyError:
                pass

            if isinstance(value, np.ndarray):  # write arrays as datasets
                self._obj.create_dataset(key, data=value)
            else:  # any other type as attribute
                self._obj.attrs[key] = value

        # also send the updated parameter to the SQL database
        self._simulation.send_to_sql(parameters={key: value})

    @mutable_only
    def update(self: Parameters[Mutable], update_dict: dict) -> None:
        """Update the parameters dictionary. This method pushes the update to
        the HDF5 file, and the SQL database.

        Args:
            update_dict: new parameters
        """
        # flatten dictionary
        update_dict = utilities.flatten_dict(update_dict)

        # update dictionary in memory
        self._dict.update(update_dict)

        # try update the sql database
        self._simulation.send_to_sql(parameters=update_dict)

        # Filter out numpy arrays
        arrays = {}
        attributes = {}
        for k, v in update_dict.items():
            if isinstance(v, np.ndarray):
                arrays[k] = update_dict.get(k)
            else:
                attributes[k] = update_dict.get(k)

        with self._file.open(FileMode.APPEND):
            # write arrays as datasets
            for k, v in arrays.items():
                if k in self._obj:
                    del self._obj[k]
                self._obj.create_dataset(k, data=v)

            # write the rest
            self._obj.attrs.update(attributes)


class Links(GroupDict[_MT]):
    def __init__(self, simulation: _Simulation) -> None:
        super().__init__(cast(HDF5File[_MT], simulation._file), "/links")


class Metadata(GroupDict[_MT]):
    _simulation: _Simulation

    def __init__(self, simulation: _Simulation) -> None:
        super().__init__(cast(HDF5File[_MT], simulation._file), "/")
        self._simulation = simulation

    @mutable_only
    def __setitem__(self: Metadata[Mutable], key: str, value: Any) -> None:
        from datetime import datetime

        # assert (
        #     key in self._dict
        # ), f'Can only set existing metadata keys. "{key}" not found.'
        if isinstance(value, datetime):
            value = value.isoformat()
        GroupDict.__setitem__(self, key, value)

        # also send the updated parameter to the SQL database
        self._simulation.send_to_sql(metadata={key: value})  # type: ignore

    @mutable_only
    def update(self: Metadata[Mutable], update_dict: _SimulationMetadataT) -> None:
        """Update the metadata dictionary. This method pushes the update to the
        HDF5 file, and the SQL database.

        Args:
            update_dict: new metadata
        """
        from datetime import datetime

        # update dictionary in memory and hdf5 file
        dict_stringified = {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in update_dict.items()
        }
        GroupDict.update(self, dict_stringified)  # type: ignore

        # try update the sql database
        self._simulation.send_to_sql(metadata=update_dict)
