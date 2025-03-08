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

from bamboost import BAMBOOST_LOGGER, constants
from bamboost._typing import _MT, Mutable, StrPath
from bamboost.constants import (
    DEFAULT_MESH_NAME,
    PATH_DATA,
    PATH_FIELD_DATA,
    PATH_MESH,
    PATH_SCALAR_DATA,
)
from bamboost.core.hdf5.file import (
    FileMode,
    H5Object,
    HDF5Path,
    WriteInstruction,
    add_to_file_queue,
    mutable_only,
)
from bamboost.core.hdf5.ref import Dataset, Group
from bamboost.core.simulation import CellType, FieldType

if TYPE_CHECKING:
    import pandas as pd

    from bamboost.core.simulation.base import _Simulation

log = BAMBOOST_LOGGER.getChild(__name__)


class GroupData(Group[_MT]):
    def __init__(self, simulation: "_Simulation"):
        super().__init__(PATH_DATA, simulation._file)
        self._simulation = simulation

    @property
    def last_step(self) -> int:
        return self.attrs.get("last_step", -1)

    @cached_property
    def fields(self) -> GroupFieldData[_MT]:
        return GroupFieldData(self)

    @cached_property
    def scalars(self) -> GroupScalarData[_MT]:
        return GroupScalarData(self)

    @property
    def timesteps(self) -> np.ndarray:
        return self[constants.DS_NAME_TIMESTEPS][:]

    @mutable_only
    def create_step(
        self: GroupData[Mutable],
        time: float = np.nan,
        step: Optional[int] = None,
    ) -> StepWriter:
        self._file._comm.barrier()
        if step is None:
            step = self.last_step + 1
        self.attrs["last_step"] = step

        # store the timestep if given
        self._store_time(step, time)

        return StepWriter(self, step)

    @mutable_only
    @add_to_file_queue
    def _store_time(self: GroupData[Mutable], step: int, time: float) -> None:
        self.require_self()

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
        timesteps = timesteps or self.timesteps

        def _create_xdmf():
            xdmf = XDMFWriter(self._file)
            xdmf.add_mesh(self._simulation.meshes[mesh_name])
            xdmf.add_timeseries(timesteps, fields, mesh_name)
            xdmf.write_file(filename)
            log.debug(f"produced XDMF file at {filename}")

        self.post_write_instruction(_create_xdmf)


class StepWriter(H5Object[Mutable]):
    def __init__(self, data_group: GroupData[Mutable], step: int):
        super().__init__(data_group._file)
        self._data_group = data_group
        self._step = step

        self.fields = data_group.fields
        self.scalars = data_group.scalars

    def add_field(
        self,
        name: str,
        data: np.ndarray,
        *,
        mesh_name: str = DEFAULT_MESH_NAME,
        field_type: FieldType = FieldType.NODE,
    ) -> None:
        field = self.fields[name]
        field.require_self()
        field.add_numerical_dataset(
            str(self._step),
            data,
            file_map=False,
            attrs={"mesh": mesh_name, "type": field_type.value},
        )
        log.debug(f"Added field {name} for step {self._step}")

    def add_fields(
        self,
        fields: dict[str, np.ndarray],
        mesh_name: str = DEFAULT_MESH_NAME,
        field_type: FieldType = FieldType.NODE,
    ) -> None:
        for name, data in fields.items():
            self.add_field(name, data, mesh_name=mesh_name, field_type=field_type)

    def add_scalar(self, name: str, data: Union[int, float, Iterable]) -> None:
        @dataclass
        class AddScalarInstruction(WriteInstruction):
            group: GroupData
            name: str
            data_arr: np.ndarray
            step: int

            def __call__(self):
                log.debug(f"Adding scalar {self.name} for step {self.step}")
                dataset = self.group.scalars.require_dataset(
                    self.name,
                    shape=(1, *self.data_arr.shape),
                    dtype=self.data_arr.dtype,
                    maxshape=(None, *self.data_arr.shape),
                    chunks=True,
                    fillvalue=np.full_like(self.data_arr, np.nan),
                )
                dataset.resize(self.step + 1, axis=0)
                dataset[self.step] = self.data_arr

        self.post_write_instruction(
            AddScalarInstruction(self._data_group, name, np.array(data), self._step)
        )

    def add_scalars(self, scalars: dict[str, Union[int, float, Iterable]]) -> None:
        for name, data in scalars.items():
            self.add_scalar(name, data)


class GroupFieldData(Group[_MT]):
    def __init__(self, data_group: GroupData[_MT]):
        super().__init__(PATH_FIELD_DATA, data_group._file)

        self._data_group = data_group
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
    _field: GroupFieldData[_MT]
    name: str

    def __new__(cls, field: GroupFieldData[_MT], name: str):
        if name not in field._field_instances:
            instance = super().__new__(cls)

            # Initialize the instance
            super(FieldData, instance).__init__(
                HDF5Path(PATH_FIELD_DATA).joinpath(name), field._file
            )
            instance._field = field
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
                        f"Index ({step_positive}) out of range for (0-{self._field._data_group.last_step})"
                    )
            else:
                return np.array([self._obj[i][rest] for i in self._slice_step(step)])  # type: ignore

    def _handle_negative_index(self, index: int) -> int:
        if index < 0:
            return (self._field._data_group.last_step or 0) + index + 1
        return index

    def _slice_step(self, step: slice) -> list[str]:
        indices = [
            str(i)
            for i in range(*step.indices((self._field._data_group.last_step or 0) + 1))
        ]
        return indices


class GroupScalarData(Group[_MT]):
    def __init__(self, data_group: GroupData[_MT]):
        super().__init__(PATH_SCALAR_DATA, data_group._file)

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
