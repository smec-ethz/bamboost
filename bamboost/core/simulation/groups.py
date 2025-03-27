from __future__ import annotations

from subprocess import CalledProcessError
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from bamboost import BAMBOOST_LOGGER
from bamboost._typing import _MT, Mutable, StrPath
from bamboost.constants import DEFAULT_MESH_NAME, PATH_MESH
from bamboost.core.hdf5.file import FileMode
from bamboost.core.hdf5.ref import Group
from bamboost.core.simulation import CellType
from bamboost.mpi import Communicator

if TYPE_CHECKING:
    from bamboost.core.simulation.base import _Simulation

log = BAMBOOST_LOGGER.getChild(__name__)


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
        res = ""
        if Communicator._active_comm.rank == 0:
            try:
                res = subprocess.run(
                    ["git", "-C", str(repo_path), *command.split()],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            except CalledProcessError:
                res = "Git command has failed"
        return Communicator._active_comm.bcast(res, root=0)

    return {
        "origin": run_git_command("remote get-url origin"),
        "commit": run_git_command("rev-parse HEAD"),
        "branch": run_git_command("rev-parse --abbrev-ref HEAD"),
        "patch": run_git_command("diff HEAD"),
    }


class GroupGit(Group[_MT]):
    def __init__(self, simulation: "_Simulation[_MT]"):
        super().__init__(".git", simulation._file)

    def add(self: GroupGit[Mutable], repo_name: str, repo_path: StrPath) -> None:
        # Make sure the .git group exists
        self.require_self()

        status = get_git_status(repo_path)
        if repo_name in self.keys():  # delete if already exists
            del self[repo_name]

        new_grp = self.require_group(repo_name)
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
