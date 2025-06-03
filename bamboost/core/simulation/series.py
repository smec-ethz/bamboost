"""Module for handling time series data in HDF5 files. The notion of time is used to
describe Series, but it can be used for any data that changes with a single parameter.

The default time series is stored at `/data` in the HDF5 file. You can create additional
series' with `bamboost.core.simulation.base.Simulation.create_series`.

This module provides classes for managing time series data stored in HDF5 files:
- Series: Main class for managing a series, including fields and global values
- FieldData: Handles a particular field (e.g., nodal or element data) and its timesteps
- GlobalData: Manages the global data that varies with time. This refers to data that is
  not tied to a mesh.
- StepWriter: Helper class for writing data at specific timesteps

The data is organized hierarchically in the HDF5 file with separate sections for
field data and global/scalar data. Fields typically represent spatial data like
displacements or stresses, while globals are used for scalar quantities like
energy or convergence metrics.
"""

from __future__ import annotations

import pkgutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

import h5py
import numpy as np

import bamboost
from bamboost import BAMBOOST_LOGGER, constants
from bamboost._typing import _MT, ArrayLike, Mutable
from bamboost.constants import (
    DEFAULT_MESH_NAME,
    PATH_DATA,
    RELATIVE_PATH_FIELD_DATA,
    RELATIVE_PATH_SCALAR_DATA,
)
from bamboost.core.hdf5.file import (
    FileMode,
    H5Object,
    WriteInstruction,
    mutable_only,
    with_file_open,
)
from bamboost.core.hdf5.hdf5path import HDF5Path
from bamboost.core.hdf5.ref import Dataset, Group, H5Reference, InvalidReferenceError
from bamboost.core.simulation import FieldType

if TYPE_CHECKING:
    import pandas as pd

    from bamboost.core.simulation.base import _Simulation

log = BAMBOOST_LOGGER.getChild(__name__)


class NotASeriesError(ValueError):
    def __init__(self, path: str):
        super().__init__(f"Path {path} exists, but is not a Series.")


class Series(H5Reference[_MT]):
    _obj: h5py.Group

    def __init__(self, simulation: _Simulation[_MT], path: str = PATH_DATA):
        super().__init__(path, simulation._file)

        # if this is not tagged a series, we raise an error
        if not self.attrs.get(".series"):
            raise NotASeriesError(path)
        self._field_instances: dict[str, FieldData[_MT]] = {}

    def __len__(self) -> int:
        try:
            return self._values().shape[0]
        except (InvalidReferenceError, KeyError):
            return 0

    @overload
    def __getitem__(self, key: tuple[()]) -> list[FieldData[_MT]]: ...
    @overload
    def __getitem__(
        self, key: Union[list[str], tuple[str], set[str]]
    ) -> list[FieldData[_MT]]: ...
    @overload
    def __getitem__(self, key: str) -> FieldData[_MT]: ...
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_field(key)
        if isinstance(key, tuple) and len(key) == 0:
            return self.get_fields()
        # else the key is a iterable of strings
        return [self.get_field(k) for k in key]

    def _ipython_key_completions_(self) -> list[HDF5Path]:
        return self.get_field_names()

    @with_file_open(FileMode.READ)
    def _repr_html_(self):
        """Repr showing the content of the group."""
        from jinja2 import Template

        html_template = pkgutil.get_data(
            bamboost.__name__, "_repr/series.html"
        ).decode()  # type: ignore
        icon = pkgutil.get_data(bamboost.__name__, "_repr/icon.txt").decode()  # type: ignore
        template = Template(html_template)

        return template.render(
            uid=Path(self._file._filename).parent.name,
            name=self._path,
            icon=icon,
            version=bamboost.__version__,
            attrs=self.attrs,
            globals=[(k, self.globals[k].shape) for k in self.globals.keys()],
            fields=self.get_field_names(),
            size=len(self),
        )

    @property
    def last_step(self) -> Union[int, None]:
        if not hasattr(self, "_last_step"):
            try:
                self._last_step = self._values().shape[0] - 1
            except KeyError:
                self._last_step = None
        return self._last_step

    @last_step.setter
    def last_step(self, value: int):
        self._last_step = value

    def get_field_names(self) -> list[HDF5Path]:
        """Return all the name of the fields in the series."""
        if not hasattr(self, "__fields_group"):
            self.__fields_group = Group(
                self._path.joinpath(RELATIVE_PATH_FIELD_DATA), self._file
            )
        return list(self.__fields_group._group_map.children())

    def get_field(self, name: str) -> FieldData[_MT]:
        """Get a field by name.

        Args:
            name: The name of the field.
        """
        if name not in self._field_instances:
            self._field_instances[name] = FieldData(self, name)
        return self._field_instances[name]

    def get_fields(self, *glob: str) -> list[FieldData[_MT]]:
        """Get multiple fields by name or glob pattern. If no arguments are given, all
        fields are returned.

        Args:
            glob: A list of glob patterns to filter the field names.
        """
        if not glob:
            return [self.get_field(name) for name in self.get_field_names()]

        import fnmatch

        matching_fields = set()
        for g in glob:
            matching_fields.update(fnmatch.filter(self.get_field_names(), g))
        return [self.get_field(name) for name in matching_fields]

    @cached_property
    def globals(self) -> GlobalData[_MT]:
        return GlobalData(self)

    @property
    @with_file_open(FileMode.READ)
    def values(self) -> np.ndarray:
        """Return the values of the series. In the default time series, this returns the
        time values of the steps."""
        try:
            return self._values()[:]
        except InvalidReferenceError:
            return np.array([])

    def _values(self) -> Dataset[_MT]:
        return super().__getitem__((constants.DS_NAME_TIMESTEPS, Dataset))

    @mutable_only
    def require_step(
        self: Series[Mutable],
        value: float = np.nan,
        step: Optional[int] = None,
    ) -> StepWriter:
        """Create a new step in the series. If the step is not given, we append one.

        Args:
            value: The value of the step. This is typically the time value.
            step: The step number. If not given, a step is appended after the last one.
        """
        if step is None:
            step = self.last_step + 1 if self.last_step is not None else 0

        # store the timestep if given
        with self.suspend_immediate_write():
            self._store_value(step, value)
            self.last_step = step

        return StepWriter(self, step)

    @mutable_only
    def _store_value(self: Series[Mutable], step: int, time: float) -> None:
        def _write_instruction():
            # require the dataset for the timesteps
            dataset = self._obj.require_dataset(
                constants.DS_NAME_TIMESTEPS,
                shape=(step + 1,),
                dtype=np.float64,
                chunks=True,
                maxshape=(None,),
                fillvalue=np.nan,
            )

            # resize the dataset and store the time
            new_size = max(step + 1, dataset.shape[0])
            log.debug(f"Resizing dataset {dataset.name} to {new_size}")
            dataset.resize(new_size, axis=0)

            log.debug(f"Storing timestep {time} for step {step}")
            dataset[step] = time

        self.post_write_instruction(_write_instruction)


class StepWriter(H5Object[Mutable]):
    """A class to write data for a specific step in a series.

    Args:
        series: The series to which the step belongs.
        step: The step number.
    """

    def __init__(self, series: Series[Mutable], step: int):
        super().__init__(series._file)
        self._series = series
        """The series to which the step belongs."""
        self._step = step
        """The step number."""

    def add_field(
        self,
        name: str,
        data: ArrayLike,
        *,
        mesh_name: str = DEFAULT_MESH_NAME,
        field_type: FieldType = FieldType.NODE,
    ) -> None:
        """Add a field to the step.

        Args:
            name: The name of the field.
            data: The data for the field.
            mesh_name: The name of the mesh to which the field belongs.
            field_type: The type of the field (default: FieldType.NODE). This is only
                relevant for XDMF writing.
        """
        field = self._series.get_field(name)
        field.require_self()
        field.add_numerical_dataset(
            str(self._step),
            data,
            file_map=True,
            attrs={"mesh": mesh_name, "type": field_type.value},
        )
        log.debug(f"Added field {name} for step {self._step}")

    def add_fields(
        self,
        fields: dict[str, ArrayLike],
        mesh_name: str = DEFAULT_MESH_NAME,
        field_type: FieldType = FieldType.NODE,
    ) -> None:
        """Add multiple fields to the step.

        Args:
            fields: A dictionary of field names and their data.
            mesh_name: The name of the mesh to which the fields belong.
            field_type: The type of the fields (default: FieldType.NODE). This is only
                relevant for XDMF writing.
        """
        with self._file.open(FileMode.APPEND, driver="mpio"):
            for name, data in fields.items():
                self.add_field(name, data, mesh_name=mesh_name, field_type=field_type)

    def add_scalar(self, name: str, data: Union[int, float, Iterable]) -> None:
        """Add a scalar to the step. Scalar data is typically a single value or a small
        array. The shape must be consistent across all steps.

        Args:
            name: The name of the scalar.
            data: The data for the scalar.

        Raises:
            ValueError: If the shape of the data is not consistent with the existing data.
        """

        @dataclass
        class AddScalarInstruction(WriteInstruction):
            series: Series
            name: str
            data_arr: np.ndarray
            step: int

            def __call__(self):
                log.info(f"Adding scalar {self.name} for step {self.step}")
                dataset = self.series.globals.require_dataset(
                    self.name,
                    shape=(1, *self.data_arr.shape),
                    dtype=float,
                    maxshape=(None, *self.data_arr.shape),
                    chunks=True,
                    fillvalue=np.nan,
                )
                new_size = max(self.step + 1, dataset.shape[0])
                if new_size > dataset.shape[0]:
                    log.info(f"Resizing dataset {dataset.name} to {new_size}")
                dataset.resize(new_size, axis=0)
                dataset[self.step] = self.data_arr

        self.post_write_instruction(
            AddScalarInstruction(self._series, name, np.array(data), self._step)
        )

    def add_scalars(self, scalars: dict[str, Union[int, float, Iterable]]) -> None:
        """Add multiple scalars to the step. See `add_scalar` for more information.

        Args:
            scalars: A dictionary of scalar names and their data.
        """
        with self.suspend_immediate_write():
            for name, data in scalars.items():
                self.add_scalar(name, data)

        self._file.single_process_queue.apply()


class FieldData(Group[_MT]):
    _parent: Series[_MT]
    name: str

    def __init__(self, series: Series[_MT], name: str):
        super().__init__(
            series._path.joinpath(RELATIVE_PATH_FIELD_DATA, name), series._file
        )
        self._parent = series
        self.name = name

    def __getitem__(
        self, key: Union[int, slice, tuple[slice | int, ...]]
    ) -> np.ndarray:
        """Get data of the field for a specific step or steps using standard slice
        notation. First index is the step number.
        """
        if isinstance(key, Iterable):
            step = key[0]
            rest = key[1:]
        else:
            step = key
            rest = ()

        with self._file.open(FileMode.READ):
            if isinstance(step, int):
                step_positive = self._handle_negative_index(step)
                try:
                    return self._obj[str(step_positive)][rest]  # type: ignore
                except KeyError:
                    raise IndexError(
                        f"Index ({step_positive}) out of range for (0-{self._parent.last_step})"
                    )
            else:
                try:
                    return np.array(
                        [self._obj[i][rest] for i in self._slice_step(step)]  # pyright: ignore[reportIndexIssue]
                    )
                except ValueError:  # The data has an inhomogeneous shape (different shape at different steps)
                    return np.array(
                        [self._obj[i][rest] for i in self._slice_step(step)],  # pyright: ignore[reportIndexIssue]
                        dtype=object,
                    )

    def at(self, step: int) -> Dataset[_MT]:
        """Get the dataset for a specific step without reading the data itself.

        Args:
            step: The step number.
        """
        step_positive = self._handle_negative_index(step)
        try:
            return self.new(
                self._path.joinpath(str(step_positive)), self._file, Dataset
            )
        except KeyError:
            raise IndexError(
                f"Index ({step_positive}) out of range for (0-{self._parent.last_step})"
            )

    def _handle_negative_index(self, index: int) -> int:
        if index < 0:
            return (self._parent.last_step or 0) + index + 1
        return index

    def _slice_step(self, step: slice) -> list[str]:
        indices = [
            str(i) for i in range(*step.indices((self._parent.last_step or 0) + 1))
        ]
        return indices


class GlobalData(Group[_MT]):
    def __init__(self, series: Series[_MT]):
        super().__init__(series._path.joinpath(RELATIVE_PATH_SCALAR_DATA), series._file)
        self._series = series

    def __getitem__(self, key: str) -> Dataset[_MT]:
        return super().__getitem__((key, Dataset[_MT]))

    @property
    def df(self) -> pd.DataFrame:
        from pandas import DataFrame

        return DataFrame(
            {"values": self._series.values} | {k: list(v[:]) for k, v in self.items()}
        )

    def get(self, *glob: str) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
        """Get the data for many scalars by name or glob pattern. If no arguments are
        given, all scalars are read and returned.

        Args:
            glob: A list of glob patterns to filter the scalar names.
        """
        if not glob:
            dataset_names = self.datasets()
            return tuple(self.__getitem__(name).array for name in dataset_names), tuple(
                map(str, dataset_names)
            )

        import fnmatch

        matching_datasets = set()
        for g in glob:
            matching_datasets.update(fnmatch.filter(self.datasets(), g))
        matching_datasets = tuple(matching_datasets)
        return tuple(
            self.__getitem__(name).array for name in matching_datasets
        ), matching_datasets
