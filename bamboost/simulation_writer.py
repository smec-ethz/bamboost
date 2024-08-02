# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import datetime
import os
import shutil
from typing import Any, Dict, Union

import numpy as np
from typing_extensions import deprecated

from bamboost import BAMBOOST_LOGGER
from bamboost.common.git_utility import GitStateGetter
from bamboost.common.mpi import MPI
from bamboost.common.utilities import flatten_dict
from bamboost.simulation import Simulation

__all__ = ["SimulationWriter"]

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


class SimulationWriter(Simulation):
    """The SimulationWriter is the writer object for a single simulation. It inherits
    all reading methods from :class:`Simulation`.

    This class can be used as a context manager. When entering the context, the status
    of the simulation is changed to "Started". When an exception is raised inside the
    context, the status is changed to "Failed [Exception]".

    Args:
        uid: The identifier of the simulation
        path: The (parent) database path
        comm: An MPI communicator (Default: `MPI.COMM_WORLD`)
    """

    def __init__(
        self,
        uid: str,
        path: str,
        comm: MPI.Comm = MPI.COMM_WORLD,
        create_if_not_exists: bool = True,
    ):
        super().__init__(uid, path, comm, create_if_not_exists)
        self.step: int = 0

    def __enter__(self):
        self.change_status("Started")  # change status to running (process 0 only)
        self._comm.barrier()  # wait for change status to be written
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.change_status(f"Failed [{exc_type.__name__}]")
        self._comm.barrier()

    def initialize(self) -> SimulationWriter:
        """Create a new file for this simlation.
        This deletes an existing h5 file of the simulation and creates an empty new one
        """
        self.step = 0
        self.add_metadata()
        self.change_status("Initiated")

        return self

    def add_metadata(self) -> None:
        """Add metadata to h5 file."""
        nb_proc = self._comm.Get_size()
        if self._prank == 0:
            with self._file("a"):
                data = {
                    "time_stamp": str(datetime.datetime.now().replace(microsecond=0)),
                    "id": self.uid,
                    "processors": nb_proc,
                    "notes": self._file.attrs.get("notes", ""),
                }
                self._file.attrs.update(data)
            self._push_update_to_sqlite(data)

    def add_parameters(self, parameters: dict) -> None:
        """Add parameters to simulation.

        Args:
            parameters: Dictionary with parameters.
        """
        if self._prank == 0:
            with self._file("a"):
                # flatten parameters
                parameters = flatten_dict(parameters)

                if "parameters" in self._file.keys():
                    del self._file["parameters"]
                grp = self._file.create_group("/parameters")
                for key, val in parameters.items():
                    if isinstance(val, np.ndarray):
                        grp.create_dataset(key, data=val)
                    elif val is not None:
                        grp.attrs[key] = val
            self._push_update_to_sqlite(parameters)

    def add_mesh(
        self, coordinates: np.ndarray, connectivity: np.ndarray, mesh_name: str = None
    ) -> None:
        """Add the mesh to file. Currently only 2d meshes.

        Args:
            coordinates: Coordinates as array (nb_nodes, dim)
            connectivity: Connectivity matrix (nb_cells, nb nodes per cell)
            mesh_name: name for mesh (default = `mesh`)
        """
        if mesh_name is None:
            mesh_name = self._default_mesh
        # self._mesh_location = 'Mesh/0/mesh/'
        mesh_location = f"{self._mesh_location}/{mesh_name}/"

        nb_nodes_local = coordinates.shape[0]
        nb_cells_local = connectivity.shape[0]

        # gather total mesh
        nb_nodes_p = np.array(self._comm.allgather(nb_nodes_local))
        nb_cells_p = np.array(self._comm.allgather(nb_cells_local))
        nb_nodes, nb_cells = np.sum(nb_nodes_p), np.sum(nb_cells_p)

        # shape of datasets
        coord_shape = (
            (nb_nodes, coordinates.shape[1]) if coordinates.ndim > 1 else (nb_nodes,)
        )
        conn_shape = (
            (nb_cells, connectivity.shape[1]) if connectivity.ndim > 1 else (nb_cells,)
        )

        # global indices nodes
        idx_start = np.sum(nb_nodes_p[self._ranks < self._prank])
        idx_end = idx_start + nb_nodes_local

        # global indices cells
        idx_start_cells = np.sum(nb_cells_p[self._ranks < self._prank])
        idx_end_cells = idx_start_cells + nb_cells_local
        connectivity = connectivity + idx_start

        with self._file("a", driver="mpio", comm=self._comm) as f:
            if mesh_location in self._file.file_object:
                del self._file.file_object[mesh_location]
            grp = f.require_group(mesh_location)
            coord = grp.require_dataset(
                "geometry", shape=coord_shape, dtype=coordinates.dtype
            )
            conn = grp.require_dataset(
                "topology", shape=conn_shape, dtype=connectivity.dtype
            )

            coord[idx_start:idx_end] = coordinates
            conn[idx_start_cells:idx_end_cells] = connectivity

    def add_field(
        self,
        name: str,
        vector: np.array,
        time: float = None,
        mesh: str = None,
        dtype: str = None,
        center: str = "Node",
    ) -> None:
        """Add a dataset to the file. The data is stored at `data/`.

        Args:
            name: Name for the dataset
            vector: Dataset
            time: Optional. time
            mesh: Optional. Linked mesh for this data
            dtype: Optional. Numpy style datatype, see h5py documentation,
                defaults to the dtype of the vector.
            center: Optional. Center of the data. Can be 'Node' or 'Cell'.
                Default is 'Node'.
        """
        if mesh is None:
            mesh = self._default_mesh

        dim = vector.shape[1:] if vector.ndim > 1 else None

        if time is None:
            time = self.step

        length_local = vector.shape[0]
        length_p = np.array(self._comm.allgather(length_local))

        length = np.sum(length_p)

        # global indices
        idx_start = np.sum(length_p[self._ranks < self._prank])
        idx_end = idx_start + length_local

        # open file
        with self._file("a", driver="mpio", comm=self._comm) as f:
            data = f.require_group(
                "data"
            )  # Require group data to store all point data in
            grp = data.require_group(name)
            vec = grp.require_dataset(
                str(self.step),
                shape=(length, *dim) if dim else (length,),
                dtype=dtype if dtype else vector.dtype,
            )
            vec[idx_start:idx_end] = vector

        if self._prank == 0:
            with self._file("a"):
                vec = self._file["data"][name][str(self.step)]
                vec.attrs["t"] = time  # add time as attribute to dataset
                vec.attrs["mesh"] = mesh  # add link to mesh as attribute
                vec.attrs["center"] = center

    def add_fields(
        self,
        fields: Dict[str, np.array],
        time: float = None,
        mesh: str = None,
    ) -> None:
        """Add multiple fields at once.

        Args:
            fields: Dictionary with fields
            time: Optional. time
        """
        for name, vector in fields.items():
            self.add_field(name, vector, time, mesh)

    def add_global_field(self, name: str, value: Any, dtype: str = None) -> None:
        """Add a gobal field. These are stored at `globals/` as an array in a
        single dataset.

        Args:
            name: Name for the data
            value: Data. Can be a numpy array or a single value.
        """
        if isinstance(value, np.ndarray):
            shape = (self.step + 1, *value.shape)
        else:
            shape = (self.step + 1,)

        if self._prank == 0:
            with self._file("a") as f:
                grp = f.require_group("globals")
                if name not in grp.keys():
                    vec = grp.create_dataset(
                        name,
                        shape=shape,
                        dtype=dtype if dtype else np.array(value).dtype,
                        chunks=True,
                        maxshape=(None, *shape[1:]) if len(shape) > 1 else (None,),
                        fillvalue=np.nan,
                    )
                    vec[-1] = value
                else:
                    vec = grp[name]
                    vec.resize(shape)
                    vec[-1] = value

    def add_global_fields(self, fields: Dict[str, Any]) -> None:
        """Add multiple global fields at once.

        Args:
            fields: Dictionary with fields
        """
        for name, value in fields.items():
            self.add_global_field(name, value)

    @deprecated("Use `copy_file` instead.")
    def add_additional(self, name: str, file: str) -> None:
        """Add an additional file stored elsewhere or in database directory.

        Args:
            name: Name of data
            file: filename of file
        """
        if self._prank == 0:
            with self._file("a") as f:
                grp = f.require_group("additionals")
                grp.attrs.update({name: file})

    def finish_step(self) -> None:
        """Finish step. Adds 1 to the step counter."""
        self.step += 1

    def finish_sim(self, status: str = "Finished") -> None:
        if self._prank == 0:
            self.change_status(status)

    def register_git_attributes(self, repo_path: str = "./") -> None:
        """Register git information for given repo.

        Args:
            repo_path (`str`): path to git repository
        """
        if self._prank == 0:
            repo_path = os.path.abspath(repo_path)
            # store current working directory
            cwd = os.getcwd()

            # switch directory to git repo
            os.chdir(repo_path)
            git_string = GitStateGetter().create_git_string()

            # switch working directory back
            os.chdir(cwd)

            with self._file("a") as f:
                grp = f.require_group("git")
                repo_name = os.path.split(repo_path)[1]
                log.info(f"Adding repo {repo_name}")
                if repo_name in grp.keys():
                    del grp[repo_name]
                grp.create_dataset(repo_name, data=git_string)

    def copy_executable(self, script_path: str) -> None:
        """WILL BE REMOVED. USE COPY_FILE.
        Copy an executable to directory for reproducability.

        Args:
            script_path: path to script
        """
        shutil.copy(script_path, self.path)
        self.executable = os.path.split(script_path)[1]

    def copy_file(self, source: Union[str, list], destination: str = "") -> None:
        """Copy a file to the datafolder.

        Args:
            source: path to file, or list of files
            destination: destination (will create intermediatory directories)
        """
        if self._prank != 0:
            return

        if isinstance(source, list):
            for item in source:
                self.copy_file(item, destination)
            return

        destination = os.path.join(self.path, destination)

        if os.path.isdir(source):
            shutil.copytree(
                source,
                os.path.join(destination, os.path.basename(source)),
                dirs_exist_ok=True,
            )
        elif os.path.isfile(source):
            os.makedirs(destination, exist_ok=True)
            shutil.copy(source, destination)
        else:
            raise FileNotFoundError
