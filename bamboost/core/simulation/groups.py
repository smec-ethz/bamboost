from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Iterable,
    Optional,
    TypedDict,
    Union,
    cast,
    overload,
)

import numpy as np
from typing_extensions import override

from bamboost import BAMBOOST_LOGGER, constants
from bamboost._typing import _MT, Mutable, StrPath
from bamboost.constants import (
    DEFAULT_MESH_NAME,
    PATH_DATA,
    PATH_MESH,
    RELATIVE_PATH_FIELD_DATA,
    RELATIVE_PATH_SCALAR_DATA,
)
from bamboost.core.hdf5.file import (
    FileMode,
    H5Object,
    WriteInstruction,
    mutable_only,
)
from bamboost.core.hdf5.ref import Dataset, Group, H5Reference
from bamboost.core.simulation import CellType, FieldType

if TYPE_CHECKING:
    import pandas as pd

    from bamboost.core.simulation.base import _Simulation

log = BAMBOOST_LOGGER.getChild(__name__)


class NotASeriesError(ValueError):
    def __init__(self, obj: H5Reference):
        super().__init__(f"Path {obj._path} exists, but is not a Series.")


class Series(Group[_MT]):
    def __init__(self, simulation: _Simulation[_MT], path: str = PATH_DATA):
        super().__init__(path, simulation._file)

        # if this is not tagged a series, we raise an error
        if not self.attrs.get(".series"):
            raise NotASeriesError(self)

        self._simulation = simulation

    @override
    def _ipython_key_completions_(self):
        return self.groups()

    @property
    def last_step(self) -> Union[int, None]:
        if not hasattr(self, "_last_step"):
            try:
                self._last_step = (
                    super().__getitem__((constants.DS_NAME_TIMESTEPS, Dataset)).shape[0]
                    - 1
                )
            except KeyError:
                self._last_step = None
        return self._last_step

    @last_step.setter
    def last_step(self, value: int):
        self._last_step = value

    @cached_property
    def fields(self) -> GroupFieldData[_MT]:
        return GroupFieldData(self)

    @cached_property
    def globals(self) -> GroupScalarData[_MT]:
        return GroupScalarData(self)

    @property
    def values(self) -> np.ndarray:
        """Return the values of the series. In the default time series, this returns the
        time values of the steps."""
        return self[constants.DS_NAME_TIMESTEPS][:]

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
            dataset = self.require_dataset(
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


class GroupFieldData(Group[_MT]):
    def __init__(self, series: Series[_MT]):
        super().__init__(series._path.joinpath(RELATIVE_PATH_FIELD_DATA), series._file)

        self._series = series
        self._field_instances: dict[str, FieldData[_MT]] = {}

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
    _parent: GroupFieldData[_MT]
    name: str

    def __new__(cls, field: GroupFieldData[_MT], name: str):
        if name not in field._field_instances:
            instance = super().__new__(cls)

            # Initialize the instance
            super(FieldData, instance).__init__(field._path.joinpath(name), field._file)
            instance._parent = field
            instance.name = name

            # Store the instance and return
            field._field_instances[name] = instance

        return field._field_instances[name]

    def __init__(self, field: GroupFieldData[_MT], name: str):
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


class GroupScalarData(Group[_MT]):
    def __init__(self, series: Series[_MT]):
        super().__init__(series._path.joinpath(RELATIVE_PATH_SCALAR_DATA), series._file)

    def __getitem__(self, key: str) -> Dataset[_MT]:
        return super().__getitem__((key, Dataset[_MT]))

    @property
    def df(self) -> pd.DataFrame:
        from pandas import DataFrame

        return DataFrame({k: list(v[:]) for k, v in self.items()})


class GroupMeshes(Group[_MT]):
    def __init__(self, simulation: "_Simulation"):
        super().__init__(PATH_MESH, simulation._file)
        self._simulation = simulation

    def __getitem__(self, key: str) -> GroupMesh[_MT]:
        return GroupMesh(self._simulation, key)

    def add(
        self: GroupMeshes[Mutable],
        nodes: np.ndarray,
        cells: np.ndarray,
        name: str = DEFAULT_MESH_NAME,
        cell_type: CellType = CellType.TRIANGLE,
    ) -> None:
        """Add a mesh with the given name to the simulation.

        Args:
            nodes: Node coordinates
            cells: Cell connectivity
            name: Name of the mesh
            cell_type: Cell type (default: "triangle"). In general, we do not care about
                the cell type and leave it up to the user to make sense of the data they
                provide. However, the cell type specified is needed for writing an XDMF
                file. For possible types, consult the XDMF/paraview manual.
        """
        with self._file.open(FileMode.APPEND, driver="mpio"):
            new_grp = self.require_group(name)
            new_grp.add_numerical_dataset("coordinates", vector=nodes)
            new_grp.add_numerical_dataset(
                "topology", vector=cells, attrs={"cell_type": cell_type.value}
            )


class GroupMesh(Group[_MT]):
    NODES = "coordinates"
    CELLS = "topology"

    def __init__(self, simulation: "_Simulation", name: str):
        super().__init__(f"{PATH_MESH}/{name}", simulation._file)

    @property
    def coordinates(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return self[self.NODES][:]

    @property
    def cells(self) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
        return self[self.CELLS][:]

    @property
    def cell_type(self) -> str:
        return self.attrs["cell_type"]


class _GitStatus(TypedDict):
    origin: str
    commit: str
    branch: str
    patch: str


def get_git_status(repo_path) -> _GitStatus:
    import subprocess

    def run_git_command(command: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_path), *command.split()],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "origin": run_git_command("remote get-url origin"),
        "commit": run_git_command("rev-parse HEAD"),
        "branch": run_git_command("rev-parse --abbrev-ref HEAD"),
        "patch": run_git_command("diff HEAD"),
    }


class GroupGit(Group[_MT]):
    def __init__(self, simulation: "_Simulation[_MT]"):
        super().__init__(".git", simulation._file)

    def add(self: GroupGit[Mutable], repo_path: StrPath) -> None:
        # Make sure the .git group exists
        self.require_self()

        status = get_git_status(repo_path)
        name = status["origin"].split("/")[-1].replace(".git", "")
        if name in self.keys():  # delete if already exists
            del self[name]

        new_grp = self.require_group(name)
        new_grp.attrs.update(
            {k: v for k, v in status.items() if k in {"origin", "commit", "branch"}}
        )
        new_grp.add_dataset("patch", data=status["patch"])

    def __getitem__(self, key: str) -> GitItem:
        grp = super().__getitem__((key, Group[_MT]))
        return GitItem(key, grp.attrs._dict, grp["patch"][()])


class GitItem:
    def __init__(self, name: str, attrs: dict[str, str], patch: bytes):
        self.name = name
        status: _GitStatus = cast(_GitStatus, attrs)
        self.branch = status["branch"]
        self.commit = status["commit"]
        self.origin = status["origin"]
        self.patch = patch.decode()

    def __repr__(self) -> str:
        return f"GitItem(name={self.name}, branch={self.branch}, commit={self.commit}, origin={self.origin}, patch={self.patch[:10]}...)"
