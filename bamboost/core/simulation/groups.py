from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Optional, Union

import numpy as np

from bamboost._typing import _MT, Mutable, StrPath
from bamboost.core.hdf5.file import FileMode, HDF5Path
from bamboost.core.hdf5.ref import Group
from bamboost.core.utilities import get_git_status

if TYPE_CHECKING:
    from bamboost.core.simulation.base import _Simulation

PATH_DATA = ".data"
PATH_FIELD_DATA = f"{PATH_DATA}/field_data"
PATH_SCALAR_DATA = f"{PATH_DATA}/scalar_data"


class GroupData(Group[_MT]):
    def __init__(self, simulation: "_Simulation"):
        super().__init__(PATH_DATA, simulation._file)
        self._simulation = simulation

    @property
    def last_step(self) -> Optional[int]:
        return self.attrs.get("last_step")

    @cached_property
    def fields(self) -> GroupFieldData[_MT]:
        return GroupFieldData(self)


class GroupFieldData(Group[_MT]):
    def __init__(self, data_group: GroupData[_MT]):
        super().__init__(PATH_FIELD_DATA, data_group._file)
        self._data_group = data_group

    def __getitem__(self, key: str) -> FieldData[_MT]:
        return FieldData(self, key)


class FieldData(Group[_MT]):
    def __init__(self, field: GroupFieldData[_MT], name: str):
        super().__init__(HDF5Path(PATH_FIELD_DATA).joinpath(name), field._file)
        self._field = field
        self.name = name

    def __getitem__(
        self, key: Union[int, slice, tuple[slice | int, ...]]
    ) -> np.ndarray:
        if isinstance(key, Iterable):
            step = key[0]
            rest = key[1:]
        else:
            step = key
            rest = ()

        with self._file.open(FileMode.READ, root_only=True):
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


class GroupGit(Group[_MT]):
    def __init__(self, simulation: "_Simulation"):
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
