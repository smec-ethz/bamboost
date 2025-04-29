from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, Generator, Mapping, cast

import h5py
import numpy as np

from bamboost import constants
from bamboost._typing import SimulationParameterT
from bamboost.core import utilities
from bamboost.core.hdf5.attrsdict import AttrsDict, AttrsEncoder, mutable_only
from bamboost.core.hdf5.file import _MT, FileMode, Mutable, with_file_open

if TYPE_CHECKING:
    from bamboost.core.simulation.base import _Simulation


class Parameters(AttrsDict[_MT]):
    _simulation: _Simulation
    _dict: SimulationParameterT

    def __init__(self, simulation: _Simulation[_MT]):
        super().__init__(simulation._file, constants.PATH_PARAMETERS)
        self._simulation = simulation

    @property
    def _obj(self) -> h5py.Group:
        obj = self._file[self._path]
        assert isinstance(obj, h5py.Group), f"Expected a group, got {type(obj)}"
        return obj

    @with_file_open(FileMode.READ)
    def read(self) -> dict:
        """Read the parameters from the HDF5 file.

        In addition to the attributes from the group, this method also reads in
        all datasets in the group.
        """
        tmp_dict = dict()

        try:
            grp = cast(h5py.Group, self._file[self._path])
        except KeyError:
            raise KeyError(
                f"Group {self._path} not found in file {self._file._filename}."
            )

        # Read in attributes
        tmp_dict.update(AttrsEncoder.decode(grp.attrs))

        # Read in datasets
        for key, value in grp.items():
            if not isinstance(value, h5py.Dataset):
                continue
            tmp_dict.update({key: value[()]})

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

        def _write_item():
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

        self.post_write_instruction(_write_item)

        # also send the updated parameter to the SQL database
        self._simulation.update_database(parameters={key: value})

    @mutable_only
    def update(self: Parameters[Mutable], update_dict: Mapping) -> None:
        """Update the parameters dictionary. This method pushes the update to
        the HDF5 file, and the SQL database.

        Args:
            update_dict: new parameters
        """
        # flatten dictionary
        flattened_dict = utilities.flatten_dict(update_dict)

        # update dictionary in memory
        self._dict.update(update_dict)

        # try update the sql database
        self._simulation.update_database(parameters=update_dict)

        # Filter out numpy arrays
        arrays = {}
        attributes = {}
        for k, v in flattened_dict.items():
            if isinstance(v, np.ndarray):
                arrays[k] = flattened_dict.get(k)
            else:
                attributes[k] = flattened_dict.get(k)

        with self._file.open(FileMode.APPEND):
            # write arrays as datasets
            for k, v in arrays.items():
                if k in self._obj:
                    del self._obj[k]
                self._obj.create_dataset(k, data=v)

            # write the rest
            self.post_write_instruction(
                lambda: self._obj.attrs.update(AttrsEncoder.encode(attributes))
            )


class Links(AttrsDict[_MT]):
    def __init__(self, simulation: _Simulation[_MT]) -> None:
        super().__init__(simulation._file, constants.PATH_LINKS)

    def __getitem__(self, key: str) -> "_Simulation":
        from bamboost.core.simulation import Simulation

        return Simulation.from_uid(super().__getitem__(key))


class Metadata(AttrsDict[_MT]):
    """The metadata of a simulation are the attributes of the root group.

    Special metadata that is handled by bamboost:
        status: Status of the simulation
        created_at: Creation date of the simulation
        description: Optional description of the simulation

    Args:
        simulation: the simulation object
    """

    _simulation: _Simulation

    def __init__(self, simulation: _Simulation[_MT]) -> None:
        super().__init__(simulation._file, "/")
        self._simulation = simulation

    @mutable_only
    def __setitem__(self: Metadata[Mutable], key: str, value: Any) -> None:
        super().__setitem__(key, value)

        # also send the updated parameter to the SQL database
        self._simulation.update_database(metadata={key: value})

    @mutable_only
    def update(self: Metadata[Mutable], update_dict: dict) -> None:
        """Update the metadata dictionary. This method pushes the update to the
        HDF5 file, and the SQL database.

        Args:
            update_dict: new metadata
        """
        # update dictionary in memory and hdf5 file
        super().update(update_dict)

        # try update the sql database
        self._simulation.update_database(metadata=update_dict)
