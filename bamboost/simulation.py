# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import os
import pkgutil
import subprocess
from contextlib import contextmanager
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Self, deprecated

from bamboost import BAMBOOST_LOGGER, index
from bamboost._config import config, paths
from bamboost.accessors.fielddata import DataGroup
from bamboost.accessors.globals import GlobalGroup
from bamboost.accessors.meshes import Mesh, MeshGroup
from bamboost.common import hdf_pointer, utilities
from bamboost.common.file_handler import FileHandler, with_file_open
from bamboost.common.mpi import MPI
from bamboost.xdmf import XDMFWriter

__all__ = [
    "Simulation",
    "Links",
]

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


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

    def __init__(
        self,
        uid: str,
        path: str,
        comm: MPI.Comm = MPI.COMM_WORLD,
        create_if_not_exists: bool = False,
        *,
        _db_id: str = None,
    ):
        self.uid: str = uid
        path = comm.bcast(path, root=0)
        self.path_database: str = os.path.abspath(path)
        self.database_id = _db_id or index.get_uid_from_path(self.path_database)
        self.path: str = os.path.abspath(os.path.join(path, uid))
        self.h5file: str = os.path.join(self.path, f"{self.uid}.h5")
        self.xdmffile: str = os.path.join(self.path, f"{self.uid}.xdmf")

        if not os.path.exists(self.h5file) and not create_if_not_exists:
            raise FileNotFoundError(
                f"Simulation {self.uid} does not exist in {self.path}."
            )

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
        self.globals: GlobalGroup = GlobalGroup(self._file, "/globals")
        self.userdata: hdf_pointer.MutableGroup = hdf_pointer.MutableGroup(
            self._file, "/userdata"
        )
        self.links: Links = Links(self._file)

    @classmethod
    def fromUID(cls, full_uid: str, *, index_database: index.IndexAPI = None) -> Self:
        """Return the `Simulation` with given UID.

        Args:
            full_uid: the full id (Database uid : simulation uid)
        """
        if index_database is None:
            index_database = index.IndexAPI()
        db_uid, sim_uid = full_uid.split(":")
        db_path = index_database.get_path(db_uid)
        return cls(sim_uid, db_path, create_if_not_exists=False)

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

    def _push_update_to_sqlite(self, update_dict: dict) -> None:
        """Push update to sqlite database.

        Args:
            - update_dict (dict): key value pair to push
        """
        if not config["options"].get("sync_table", True):
            return
        try:
            index.DatabaseTable(self.database_id).update_entry(self.uid, update_dict)
        except index.Error as e:
            log.warning(f"Could not update sqlite database: {e}")

    @property
    def parameters(self) -> dict:
        tmp_dict = dict()
        if self._prank == 0:
            with self._file("r"):
                # return if parameters is not in the file
                if "parameters" not in self._file.keys():
                    return {}
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

        tmp_dict = utilities.unflatten_dict(tmp_dict)
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

    def open_in_paraview(self) -> None:
        """Open the xdmf file in paraview."""
        subprocess.Popen(["paraview", self.xdmffile])

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

        self._push_update_to_sqlite({"status": status})

    def update_metadata(self, update_dict: dict) -> None:
        """Update the metadata attributes.

        Args:
            update_dict: dictionary to push
        """
        if self._prank == 0:
            update_dict = utilities.flatten_dict(update_dict)
            with self._file("a") as file:
                file.attrs.update(update_dict)

            self._push_update_to_sqlite(update_dict)

    def update_parameters(self, update_dict: dict) -> None:
        """Update the parameters dictionary.

        Args:
            update_dict: dictionary to push
        """
        if self._prank == 0:
            update_dict = utilities.flatten_dict(update_dict)
            with self._file("a") as file:
                file["parameters"].attrs.update(update_dict)

            self._push_update_to_sqlite(update_dict)

    def create_xdmf_file(self, fields: list = None, nb_steps: int = None) -> None:
        """Create the xdmf file to read in paraview.

        Args:
            fields (list[str]): fields for which to write timeseries information,
                if not specified, all fields in data are written.
            nb_steps (int): number of steps the simulation has
        """

        if self._prank == 0:
            with self._file("r") as f:
                if "data" not in f.keys():
                    fields, nb_steps = [], 0
                if fields is None:
                    fields = list(f["data"].keys())

                if nb_steps is None:
                    grp_name = list(f["data"].keys())[0]
                    nb_steps = list(f[f"data/{grp_name}"].keys())
                    nb_steps = max(
                        [
                            int(step)
                            for step in nb_steps
                            if not (
                                step.startswith("__") or step.endswith("_intermediates")
                            )
                        ]
                    )

                # temporary fix to load coordinates/geometry
                coords_name = (
                    "geometry"
                    if "geometry"
                    in f[f"{self._mesh_location}/{self._default_mesh}"].keys()
                    else "coordinates"
                )

            with self._file("r"):
                xdmf_writer = XDMFWriter(self.xdmffile, self._file)
                xdmf_writer.write_points_cells(
                    f"{self._mesh_location}/{self._default_mesh}/{coords_name}",
                    f"{self._mesh_location}/{self._default_mesh}/topology",
                )

                if fields:
                    xdmf_writer.add_timeseries(nb_steps + 1, fields)
                xdmf_writer.write_file()

        self._comm.barrier()

    def create_run_script(
        self,
        commands: list[str],
        euler: bool = True,
        sbatch_kwargs: dict[str, Any] = None,
    ) -> None:
        """Create a batch job and put it into the folder.

        Args:
            commands: A list of strings being the user defined commands to run
            euler: If false, a local bash script will be written
            sbatch_kwargs: Additional sbatch arguments.
                This parameter allows you to provide additional arguments to the `sbatch` command
                when submitting jobs to a Slurm workload manager. The arguments should be provided
                in the format of a dict of sbatch option name and values.

                Use this parameter to specify various job submission options such as the number of
                tasks, CPU cores, memory requirements, email notifications, and other sbatch options
                that are not covered by default settings.
                By default, the following sbatch options are set:
                - `--output`: The output file is set to `<uid>.out`.
                - `--job-name`: The job name is set to `<full_uid>`.

                The following arguments should bring you far:
                - `--ntasks`: The number of tasks to run. This is the number of MPI processes to start.
                - `--mem-per-cpu`: The memory required per CPU core.
                - `--time`: The maximum time the job is allowed to run.
                - `--tmp`: Temporary scratch space to use for the job.
        """
        def _set_environment_variables():
            return (
                f"""DATABASE_DIR=$(sqlite3 {paths['DATABASE_FILE']} "SELECT path FROM dbindex WHERE id='{self.database_id}'")\n"""
                f"SIMULATION_DIR=$DATABASE_DIR/{self.uid}\n"
                f"SIMULATION_ID={self.database_id}:{self.uid}\n\n"
            )

        script = "#!/bin/bash\n\n"
        if euler:
            # Write sbatch submission script for EULER
            # filename: sbatch_{self.uid}.sh
            if sbatch_kwargs is None:
                sbatch_kwargs = {}

            sbatch_kwargs.setdefault("--output", f"{self.path}/{self.uid}.out")
            sbatch_kwargs.setdefault("--job-name", self.get_full_uid())

            for key, value in sbatch_kwargs.items():
                script += f"#SBATCH {key}={value}\n"

            script += "\n"
            script += _set_environment_variables()
            script += "\n".join(commands)

            with open(self.files(f"sbatch_{self.uid}.sh"), "w") as file:
                file.write(script)
        else:
            # Write local bash script
            # filename: {self.uid}.sh
            script += _set_environment_variables()
            script += "\n".join(commands)

            with open(self.files(f"{self.uid}.sh"), "w") as file:
                file.write(script)

        with self._file("a") as file:
            file.attrs.update({"submitted": False})
        self._push_update_to_sqlite({"submitted": False})

    @deprecated("use `create_run_script` instead")
    def create_batch_script(self, *args, **kwargs):
        return self.create_run_script(*args, **kwargs)

    def submit(self) -> None:
        """Submit the job for this simulation."""
        if f"sbatch_{self.uid}.sh" in os.listdir(self.path):
            batch_script = os.path.abspath(
                os.path.join(self.path, f"sbatch_{self.uid}.sh")
            )
            env = os.environ.copy()
            _ = env.pop("BAMBOOST_MPI", None)
            subprocess.run(["sbatch", f"{batch_script}"], env=env)
        elif f"{self.uid}.sh" in os.listdir(self.path):
            bash_script = os.path.abspath(os.path.join(self.path, f"{self.uid}.sh"))
            env = os.environ.copy()
            _ = env.pop("BAMBOOST_MPI", None)
            subprocess.run(["bash", f"{bash_script}"], env=env)
        else:
            raise FileNotFoundError(
                f"Could not find a batch script for simulation {self.uid}."
            )

        log.info(f"Simulation {self.uid} submitted!")

        with self._file("a") as file:
            file.attrs.update({"submitted": True})

        self._push_update_to_sqlite({"submitted": True})

    @with_file_open("a")
    def change_note(self, note) -> None:
        if self._prank == 0:
            self._file.attrs["notes"] = note
            self._push_update_to_sqlite({"notes": note})

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
    def mesh(self) -> Mesh:
        """Return the default mesh.

        Returns:
            MeshGroup
        """
        return self.meshes[self._default_mesh]

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

        # Raise an error if the mesh is not found
        if (self._mesh_location.split("/")[0] not in self._file.keys()) or (
            mesh_name not in self._file[self._mesh_location].keys()
        ):
            raise KeyError(f"Mesh location {self._mesh_location} not found in file.")

        mesh = self.meshes[mesh_name]
        return mesh.coordinates, mesh.connectivity

    @property
    @deprecated("Use `data.info` instead")
    @with_file_open("r")
    def data_info(self) -> pd.DataFrame:
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
        print("\U0001f43c " + os.path.basename(self.h5file))
        utilities.h5_tree(self._file.file_object)

    @contextmanager
    def enter_path(self):
        """A context manager for changing the working directory to this simulations' path.

        >>> with sim.working_directory():
        >>>     ...
        """

        current_dir = os.getcwd()
        try:
            os.chdir(self.path)
            yield
        finally:
            os.chdir(current_dir)
