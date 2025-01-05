from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Union

import numpy as np

from bamboost._typing import _MT, Mutable, StrPath
from bamboost.core.hdf5.file import FileMode
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

        with self._file.open(FileMode.APPEND, root_only=True):
            self._obj.require_dataset(
                "times",
                (0,),
                dtype=np.float64,
                chunks=True,
                maxshape=(None,),
                fillvalue=np.nan,
            )

    @property
    def last_step(self) -> Optional[int]:
        return self.attrs.get("last_step")


class GroupFieldData(Group[_MT]):
    def __init__(self, data_group: GroupData[_MT]):
        super().__init__(PATH_FIELD_DATA, data_group._file)
        self._data_group = data_group

    def __getitem__(self, key: str) -> FieldData[_MT]:
        return FieldData(self, key)


class FieldData(Group[_MT]):
    def __init__(self, group: GroupFieldData[_MT], name: str):
        self._group = group
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

        if isinstance(step, int):
            return (self._obj[str(step)])[rest]  # type: ignore
        else:
            return np.array([self._obj[i][rest] for i in self._slice_step(step)])  # type: ignore

    def _slice_step(self, step: slice) -> list[str]:
        indices = [
            str(i)
            for i in range(*step.indices((self._group._data_group.last_step or 0) + 1))
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
