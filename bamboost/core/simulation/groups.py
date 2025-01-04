from __future__ import annotations

from typing import TYPE_CHECKING

from bamboost._typing import _MT, Mutable, StrPath
from bamboost.core.hdf5.ref import Group
from bamboost.core.utilities import get_git_status

if TYPE_CHECKING:
    from bamboost.core.simulation.base import _Simulation


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
