# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import logging
import os
import pkgutil
import subprocess
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
from typing_extensions import Self, deprecated

from . import index
from .accessors.fielddata import DataGroup
from .accessors.globals import GlobalGroup
from .accessors.meshes import MeshGroup
from .common import hdf_pointer, utilities
from .common.file_handler import FileHandler, with_file_open
from .common.job import Job
from .xdmf import XDMFWriter

__all__ = ["Simulation", "Links"]

log = logging.getLogger(__name__)


class Links(hdf_pointer.MutableGroup):
    """Link group. Used to create and access links.

    I don't know how to distribute this to its own file in the accessors
    directory, due to circular imports.
    """

    def __init__(self, file_handler: FileHandler) -> None:
        super().__init__(file_handler, path_to_data="links")

    def _ipython_key_completions_(self):
        return tuple(self.all_links().keys())

    @with_file_open("r", driver="mpio")
    def __getitem__(self, key) -> Simulation:
        """Returns the linked simulation object."""
        return Simulation.fromUID(self.obj.attrs[key])

    def __setitem__(self, key, newvalue):
        """Creates the link."""
        return self.update_attrs({key: newvalue})

    def __delitem__(self, key):
        """Delete a link."""
        with self._file("a"):
            del self.obj.attrs[key]

    @with_file_open("r")
    def __repr__(self) -> str:
        return repr(
            pd.DataFrame.from_dict(self.all_links(), orient="index", columns=["UID"])
        )

    @with_file_open("r")
    def _repr_html_(self) -> str:
        return pd.DataFrame.from_dict(
            self.all_links(), orient="index", columns=["UID"]
        )._repr_html_()

    @with_file_open("r")
    def all_links(self) -> dict:
        return dict(self.obj.attrs)


class Simulation:
    """A single dataset/simulation. Used to write to it, read from it or append.

    Args:
        uid (str): unique identifier
        path (str): path to parent/database folder
        comm (MPI.Comm): MPI communicator (default=MPI.COMM_WORLD)
    """

    _mesh_location = "Mesh/0"
    _default_mesh = "mesh"

    def __init__(self, uid: str, path: str, comm: MPI.Comm = MPI.COMM_WORLD):
        self.uid: str = uid
        self.path_database: str = os.path.abspath(path)
        self.path: str = os.path.abspath(os.path.join(path, uid))
        self.h5file: str = os.path.join(self.path, f"{self.uid}.h5")
        self.xdmffile: str = os.path.join(self.path, f"{self.uid}.xdmf")
        os.makedirs(self.path, exist_ok=True)

        # MPI information
        self._comm = comm
        self._psize = self._comm.size
        self._prank = self._comm.rank
        self._ranks = np.array([i for i in range(self._psize)])

        self._file = FileHandler(self.h5file)

        # Initialize groups to meshes, data and userdata. Create groups.
        self.meshes: MeshGroup = MeshGroup(self._file)
        self.data: DataGroup = DataGroup(self._file, self.meshes)
        self.globals: GlobalGroup = GlobalGroup(self._file, '/globals')
        self.userdata: hdf_pointer.MutableGroup = hdf_pointer.MutableGroup(
            self._file, "/userdata"
        )
        self.links: Links = Links(self._file)

    @classmethod
    def fromUID(cls, full_uid: str) -> Self:
        """Return the `Simulation` with given UID.

        Args:
            full_uid: the full id (Database uid : simulation uid)
        """
        db_uid, sim_uid = full_uid.split(":")
        db_path = index.get_path(db_uid)
        return cls(sim_uid, db_path)

    @with_file_open()
    def __getitem__(self, key) -> hdf_pointer.BasePointer:
        """Direct access to HDF5 file.

        Returns:
            :class:`~bamboost.common.file_handler.BasePointer`
        """
        return hdf_pointer.BasePointer.new_pointer(self._file, key)

    def _repr_html_(self) -> str:
        html_string = pkgutil.get_data(__name__, "html/simulation.html").decode()
        icon = pkgutil.get_data(__name__, "html/icon.txt").decode()

        table_string = ""
        for key, value in self.parameters.items():
            if (
                isinstance(value, Iterable)
                and not isinstance(value, str)
                and len(value) > 5
            ):
                value = "..."
            table_string += f"""
            <tr>
                <td>{key}</td>
                <td>{value}</td>
            </tr>
            """

        metadata = self.metadata

        def get_pill_div(text: str, color: str):
            return (
                f'<div class="status" style="background-color:'
                f'var(--bb-{color});">{text}</div>'
            )

        status_options = {
            "Finished": get_pill_div("Finished", "green"),
            "Failed": get_pill_div("Failed", "red"),
            "Initiated": get_pill_div("Initiated", "grey"),
        }
        submitted_options = {
            True: get_pill_div("Submitted", "green"),
            False: get_pill_div("Not submitted", "grey"),
        }

        html_string = (
            html_string.replace("$UID", self.uid)
            .replace("$ICON", icon)
            .replace("$TREE", self.show_files(printit=False).replace("\n", "<br>"))
            .replace("$TABLE", table_string)
            .replace("$NOTE", metadata["notes"])
            .replace(
                "$STATUS",
                status_options.get(
                    metadata["status"],
                    f'<div class="status">{metadata["status"]}</div>',
                ),
            )
            .replace("$SUBMITTED", submitted_options[metadata.get("submitted", False)])
            .replace("$TIMESTAMP", metadata["time_stamp"])
        )
        return html_string

    @property
    def parameters(self) -> dict:
        tmp_dict = dict()
        if self._prank == 0:
            with self._file("r"):
                tmp_dict.update(self._file["parameters"].attrs)
                for key in self._file["parameters"].keys():
                    tmp_dict.update({key: self._file[f"parameters/{key}"][()]})

        tmp_dict = utilities.unflatten_dict(tmp_dict)

        tmp_dict = self._comm.bcast(tmp_dict, root=0)
        return tmp_dict

    @property
    def metadata(self) -> dict:
        tmp_dict = dict()
        if self._prank == 0:
            with self._file("r") as file:
                tmp_dict.update(file.attrs)
        tmp_dict = self._comm.bcast(tmp_dict, root=0)
        return tmp_dict

    def files(self, filename: str) -> str:
        """Get the path to the file.

        Args:
            filename: name of the file
        """
        return os.path.join(self.path, filename)

    def show_files(
        self, level=-1, limit_to_directories=False, length_limit=1000, printit=True
    ) -> str:
        """Show the file tree of the simulation directory.

        Args:
            level: how deep to print the tree
            limit_to_directories: only print directories
            length_limit: cutoff
        """
        tree_string = utilities.tree(
            self.path, level, limit_to_directories, length_limit
        )
        if printit:
            print(tree_string)
        else:
            return tree_string

    def open_in_file_explorer(self) -> None:
        """Open the simulation directory. Uses `xdg-open` on linux systems."""
        if os.name == "nt":  # should work on Windows
            os.startfile(self.path)
        else:
            subprocess.run(["xdg-open", self.path])

    def get_full_uid(self) -> str:
        """Returns the full uid of the simulation (including the one of the database)"""
        database_uid = index.get_uid_from_path(self.path_database)
        return f"{database_uid}:{self.uid}"

    def change_status(self, status: str) -> None:
        """Change status of simulation.

        Args:
            status (str): new status
        """
        if self._prank == 0:
            self._file.open("a")
            self._file.attrs["status"] = status
            self._file.close()

    def update_metadata(self, update_dict: dict) -> None:
        """Update the metadata attributes.

        Args:
            update_dict: dictionary to push
        """
        with self._file("a") as file:
            file.attrs.update(update_dict)

    def update_parameters(self, update_dict: dict) -> None:
        """Update the parameters dictionary.

        Args:
            update_dict: dictionary to push
        """
        if self._prank == 0:
            with self._file("a") as file:
                file["parameters"].attrs.update(utilities.flatten_dict(update_dict))

    def create_xdmf_file(self, fields: list = None, nb_steps: int = None) -> None:
        """Create the xdmf file to read in paraview.

        Args:
            fields (list[str]): fields for which to write timeseries information,
                if not specified, all fields in data are written.
            nb_steps (int): number of steps the simulation has
        """

        if self._prank == 0:
            with self._file("r") as f:
                if not fields:
                    fields = list(f["data"].keys())

                if not nb_steps:
                    grp_name = list(f["data"].keys())[0]
                    nb_steps = list(f[f"data/{grp_name}"].keys())
                    nb_steps = max(
                        [int(step) for step in nb_steps if not step.startswith("__")]
                    )

            xdmf_writer = XDMFWriter(self.xdmffile, self.h5file)
            xdmf_writer.write_points_cells(
                f"{self._mesh_location}/{self._default_mesh}/geometry",
                f"{self._mesh_location}/{self._default_mesh}/topology",
            )

            xdmf_writer.add_timeseries(nb_steps + 1, fields)
            xdmf_writer.write_file()

    def create_batch_script(
        self,
        commands: list = None,
        nnodes=1,
        ntasks=4,
        ncpus=1,
        time="04:00:00",
        mem_per_cpu=2048,
        tmp=8000,
        euler=True,
    ) -> None:
        """Create a batch job and put it into the folder.

        Args:
            commands: A list of strings being the user defined commands to run
            nnodes: nb of nodes (default=1)
            ntasks: nb of tasks (default=4)
            ncpus: nb of cpus per task (default=1)
            time: requested time (default=4 hours)
            mem_per_cpu: memory (default=2048)
            tmp: temporary storage, set None to exclude option (default=8000)
            euler: If false, a local bash script will be written
        """
        job = Job()
        if not commands:
            if hasattr(self, "executable"):
                if ".py" in self.executable:
                    command = (
                        f"{{MPI}} python3 {os.path.join(self.path, self.executable)} "
                        f"--path {self.path_database} --uid {self.uid}"
                    )
                    commands = [command]
            else:
                raise AttributeError(
                    """Either you must specify an executable or have it 
                                     copied before with `copy_executable`!"""
                )

        if euler:
            job.create_sbatch_script(
                commands,
                path=os.path.abspath(self.path_database),
                uid=self.uid,
                nnodes=nnodes,
                ntasks=ntasks,
                ncpus=ncpus,
                time=time,
                mem_per_cpu=mem_per_cpu,
                tmp=tmp,
            )
        else:
            job.create_bash_script_local(
                commands,
                path=os.path.abspath(self.path_database),
                uid=self.uid,
                ntasks=ntasks,
            )
        with self._file("a") as file:
            file.attrs.update({"submitted": False})

    def submit(self) -> None:
        """Submit the job for this simulation."""
        if f"sbatch_{self.uid}.sh" in os.listdir(self.path):
            batch_script = os.path.abspath(
                os.path.join(self.path, f"sbatch_{self.uid}.sh")
            )
            subprocess.Popen(["sbatch", f"{batch_script}"])
        if f"{self.uid}.sh" in os.listdir(self.path):
            bash_script = os.path.abspath(os.path.join(self.path, f"{self.uid}.sh"))
            subprocess.Popen(["bash", f"{bash_script}"])

        print(f"Simulation {self.uid} submitted!")

        with self._file("a") as file:
            file.attrs.update({"submitted": True})

    @with_file_open("a")
    def change_note(self, note) -> None:
        self._file.attrs["notes"] = note

    # Ex-Simulation reader methods
    # ----------------------------

    def open(self, mode: str = "r", driver=None, comm=None) -> FileHandler:
        """Use this as a context manager in a `with` statement.
        Purpose: keeping the file open to directly access/edit something in the
        HDF5 file of this simulation.

        Args:
            mode (`str`): file mode (see h5py docs)
            driver (`str`): file driver (see h5py docs)
            comm (`str`): mpi communicator
        """
        return self._file(mode, driver, comm)

    @property
    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return coordinates and connectivity of default mesh.

        Returns:
            Tuple of np.arrays (coordinates, connectivity)
        """
        return self.get_mesh()

    @with_file_open("r")
    def get_mesh(self, mesh_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return coordinates and connectivity. Currently returns numpy arrays.

        Args:
            mesh_name (`str`): optional, name of mesh to read (default = mesh)
        Returns:
            Tuple of np.arrays (coordinates, connectivity)
        """
        if mesh_name is None:
            mesh_name = self._default_mesh

        mesh = self.meshes[mesh_name]
        return mesh.coordinates, mesh.connectivity

    @property
    @deprecated("Use `data.info` instead")
    @with_file_open("r")
    def data_info(self) -> pd.Dataframe:
        """View the data stored.

        Returns:
            :class:`pd.DataFrame`
        """
        tmp_dictionary = dict()
        for data in self.data:
            steps = len(data)
            shape = data.obj["0"].shape
            dtype = data.obj["0"].dtype
            tmp_dictionary[data._name] = {
                "dtype": dtype,
                "shape": shape,
                "steps": steps,
            }
        return pd.DataFrame.from_dict(tmp_dictionary)

    @property
    @with_file_open("r")
    def git(self) -> dict:
        """Get Git information.

        Returns:
            :class:`dict` with different repositories.
        """
        if "git" not in self._file.keys():
            return "Sorrrry, no git information stored :()"
        grp = self._file["git"]
        tmp_dict = {}
        for repo in grp.keys():
            tmp_dict[repo] = grp[repo][()].decode("utf8")
        return tmp_dict

    def get_data_interpolator(self, field: str, step: int):
        """Get Linear interpolator for data field at step. Uses the linked mesh.

        Args:
            name (`str`): name of the data field
            step (`int`): step
        Returns:
            :class:`scipy.interpolate.LinearNDInterpolator`
        """
        from scipy.interpolate import LinearNDInterpolator

        return LinearNDInterpolator(
            self.data[field].mesh.coordinates, self.data[field].at_step(step)
        )

    @with_file_open()
    def show_h5tree(self) -> None:
        """Print the tree inside the h5 file."""
        # print('\U00002B57 ' + os.path.basename(self.h5file))
        print("\U0001F43C " + os.path.basename(self.h5file))
        utilities.h5_tree(self._file.file_object)
