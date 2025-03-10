from __future__ import annotations

import pkgutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    Optional,
    Union,
    overload,
)

import h5py
import numpy as np

import bamboost
from bamboost import BAMBOOST_LOGGER, constants
from bamboost._typing import _MT, Mutable, StrPath
from bamboost.constants import (
    DEFAULT_MESH_NAME,
    PATH_DATA,
    RELATIVE_PATH_FIELD_DATA,
    RELATIVE_PATH_SCALAR_DATA,
)
from bamboost.core.hdf5.attrsdict import AttrsDict
from bamboost.core.hdf5.file import (
    FileMode,
    H5Object,
    WriteInstruction,
    mutable_only,
    with_file_open,
)
from bamboost.core.hdf5.ref import Dataset, Group, H5Reference
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
        self.attrs = AttrsDict(self._file, path)
        if not self.attrs.get(".series"):
            raise NotASeriesError(path)

    def __len__(self) -> int:
        return self._values().shape[0]

    @with_file_open(FileMode.READ)
    def _repr_html_(self):
        """Repr showing the content of the group."""
        # If the object is not a group, return a simple representation
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
            fields=self.fields.keys(),
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

    @cached_property
    def fields(self) -> SeriesFields[_MT]:
        return SeriesFields(self)

    @cached_property
    def globals(self) -> GlobalData[_MT]:
        return GlobalData(self)

    @property
    @with_file_open(FileMode.READ)
    def values(self) -> np.ndarray:
        """Return the values of the series. In the default time series, this returns the
        time values of the steps."""
        return self._values()[:]

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

        # wait for root process to finish writing
        self._file._comm.barrier()

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

    def create_xdmf(
        self,
        field_names: Optional[Union[tuple[str], list[str], set[str]]] = None,
        timesteps: Optional[Iterable[float]] = None,
        *,
        filename: Optional[StrPath] = None,
        mesh_name: str = DEFAULT_MESH_NAME,
    ):
        from bamboost.core.simulation.xdmf import XDMFWriter

        fields = self.fields[field_names] if field_names else self.fields[()]
        filename = filename or self._simulation.path.joinpath(constants.XDMF_FILE_NAME)
        timesteps = timesteps or self.values

        def _create_xdmf():
            xdmf = XDMFWriter(self._file)
            xdmf.add_mesh(self._simulation.meshes[mesh_name])
            xdmf.add_timeseries(timesteps, fields, mesh_name)
            xdmf.write_file(filename)
            log.debug(f"produced XDMF file at {filename}")

        self.post_write_instruction(_create_xdmf)


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
        data: np.ndarray,
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
        field = self._series.fields[name]
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
        fields: dict[str, np.ndarray],
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
            group: Series
            name: str
            data_arr: np.ndarray
            step: int

            def __call__(self):
                log.debug(f"Adding scalar {self.name} for step {self.step}")
                dataset = self.group.globals.require_dataset(
                    self.name,
                    shape=(1, *self.data_arr.shape),
                    dtype=float,
                    maxshape=(None, *self.data_arr.shape),
                    chunks=True,
                    fillvalue=np.nan,
                )
                dataset.resize(self.step + 1, axis=0)
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


class SeriesFields(H5Reference[_MT]):
    def __init__(self, series: Series[_MT]):
        super().__init__(series._path.joinpath(RELATIVE_PATH_FIELD_DATA), series._file)

        self._series = series
        self._field_instances: dict[str, FieldData[_MT]] = {}

    @with_file_open(FileMode.READ)
    def keys(self) -> list[str]:
        return list(self._obj.keys())  # pyright: ignore[reportAttributeAccessIssue]

    def _ipython_key_completions_(self):
        return self.keys()

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
            return FieldData(self, key)
        if isinstance(key, tuple) and len(key) == 0:
            return [FieldData(self, k) for k in self.keys()]
        # else the key is a iterable of strings
        return [FieldData(self, k) for k in key]


class FieldData(Group[_MT]):
    _parent: SeriesFields[_MT]
    name: str

    def __new__(cls, field: SeriesFields[_MT], name: str):
        if name not in field._field_instances:
            instance = super().__new__(cls)

            # Initialize the instance
            super(FieldData, instance).__init__(field._path.joinpath(name), field._file)
            instance._parent = field
            instance.name = name

            # Store the instance and return
            field._field_instances[name] = instance

        return field._field_instances[name]

    def __init__(self, field: SeriesFields[_MT], name: str):
        # initialization is done in __new__ for simplicity
        pass

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
                        f"Index ({step_positive}) out of range for (0-{self._parent._series.last_step})"
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

    def _handle_negative_index(self, index: int) -> int:
        if index < 0:
            return (self._parent._series.last_step or 0) + index + 1
        return index

    def _slice_step(self, step: slice) -> list[str]:
        indices = [
            str(i)
            for i in range(*step.indices((self._parent._series.last_step or 0) + 1))
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
