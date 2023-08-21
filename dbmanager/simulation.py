# This file is part of dbmanager, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Tuple, Union
from types import SimpleNamespace
import numpy as np
import pandas as pd
import datetime
import logging
import h5py
from mpi4py import MPI

from .xdmf import XDMFWriter
from .common.git_utility import GitStateGetter
from .common.job import Job
from .common.file_handler import FileHandler, with_file_open
from .common.utilities import flatten_dict, unflatten_dict
from .common import hdf_pointer

log = logging.getLogger(__name__)



class Simulation:
    """A single dataset/simulation. Used to write to it, read from it or append.

    Args:
        uid (str): unique identifier
        path (str): path to parent/database folder
        comm (MPI.Comm): MPI communicator (default=MPI.COMM_WORLD)
    """
    _mesh_location = 'Mesh/0'
    _default_mesh = 'mesh'

    def __init__(self, uid: str, path: str, comm: MPI.Comm = MPI.COMM_WORLD):
        self.uid = uid
        self.path_database = os.path.abspath(path)
        self.path = os.path.abspath(os.path.join(path, uid))
        self.h5file = os.path.join(self.path, f'{self.uid}.h5')
        self.xdmffile = os.path.join(self.path, f'{self.uid}.xdmf')
        os.makedirs(self.path, exist_ok=True)

        # MPI information
        self._comm = comm
        self._psize = self._comm.size
        self._prank = self._comm.rank
        self._ranks = np.array([i for i in range(self._psize)])

        # FileHandler must be shared between processes!
        file_handler = None
        if self._prank==0:
            file_handler = FileHandler(f'{self.h5file}')
        self._file = self._comm.bcast(file_handler, root=0)

    @with_file_open()
    def __getitem__(self, key) -> hdf_pointer.BasePointer:
        """Direct access to HDF5 file.

        Returns:
            :class:`~dbmanager.common.file_handler.BasePointer`
        """
        return hdf_pointer.get_best_pointer(self._file[key])

    @property
    def parameters(self):
        tmp_dict = dict()
        if self._prank==0:
            with self._file('r'):
                tmp_dict.update(self._file['parameters'].attrs)
                for key in self._file['parameters'].keys():
                    tmp_dict.update({key: self._file[f'parameters/{key}'][()]})

        tmp_dict = unflatten_dict(tmp_dict)

        tmp_dict = self._comm.bcast(tmp_dict, root=0)
        return tmp_dict

    @property
    def metadata(self):
        tmp_dict = dict()
        if self._prank==0:
            with self._file('r') as file:
                tmp_dict.update(file.attrs)
        tmp_dict = self._comm.bcast(tmp_dict, root=0)
        return tmp_dict

    @with_file_open('a')
    def change_status(self, status: str) -> None:
        """Change status of simulation.

        Args:
            status (str): new status
        """
        if self._prank==0:
            self._file.attrs['status'] = status

    def update_metadata(self, update_dict: dict) -> None:
        """Update the metadata attributes.

        Args:
            update_dict: dictionary to push
        """
        with self._file('a') as file:
            file.attrs.update(update_dict)

    def update_parameters(self, update_dict: dict) -> None:
        """Update the parameters dictionary.

        Args:
            update_dict: dictionary to push
        """
        if self._prank==0:
            with self._file('a') as file:
                file['parameters'].attrs.update(update_dict)

    def create_xdmf_file(self, fields: list = None, nb_steps: int = None) -> None:
        """Create the xdmf file to read in paraview.

        Args:
            fields (list[str]): fields for which to write timeseries information,
                if not specified, all fields in data are written.
            nb_steps (int): number of steps the simulation has
        """

        if self._comm.rank==0:

            with self._file('r') as f:
                if not fields:
                    fields = list(f['data'].keys())

                if not nb_steps:
                    grp_name = list(f['data'].keys())[0]
                    nb_steps = list(f[f'data/{grp_name}'].keys())
                    nb_steps = max([int(step) for step in nb_steps])

            xdmf_writer = XDMFWriter(self.xdmffile, self.h5file)
            xdmf_writer.write_points_cells(f'{self._mesh_location}/{self._default_mesh}/geometry',
                                           f'{self._mesh_location}/{self._default_mesh}/topology')

            xdmf_writer.add_timeseries(nb_steps+1, fields)
            xdmf_writer.write_file()

    def create_batch_script(self, commands: list = None, nnodes=1, ntasks=4,
                         ncpus=1, time='04:00:00', mem_per_cpu=2048, tmp=8000, euler=True) -> None:
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
            if hasattr(self, 'executable'):
                if '.py' in self.executable:
                    command = (f"{{MPI}} python3 {os.path.join(self.path, self.executable)} "
                               f"--path {self.path_database} --uid {self.uid}")
                    commands = [command]
            else:
                raise AttributeError("""Either you must specify an executable or have it 
                                     copied before with `copy_executable`!""")
        
        if euler:
            job.create_sbatch_script(commands, path=os.path.abspath(self.path_database), uid=self.uid, nnodes=nnodes,
                                     ntasks=ntasks, ncpus=ncpus, time=time, mem_per_cpu=mem_per_cpu, tmp=tmp)
        else:
            job.create_bash_script_local(commands, path=os.path.abspath(self.path_database),
                                         uid=self.uid, ntasks=ntasks)
        with self._file('a') as file:
            file.attrs.update({'submitted': False})

    def submit(self) -> None:
        """Submit the job for this simulation."""
        batch_script = os.path.abspath(os.path.join(self.path, f'sbatch_{self.uid}.sh'))
        subprocess.Popen(["sbatch", f"{batch_script}"])
        print(f'Simulation {self.uid} submitted!')

        with self._file('a') as file:
            file.attrs.update({'submitted': True})

    @with_file_open('a')
    def change_note(self, note) -> None:
        self._file.attrs['notes'] = note

    @with_file_open('a')
    def add_postprocess_data(self, name: str, vector: np.array) -> None:
        """Add a custom field in postprocessing."""
        data = self._file.require_group('postprocess')  # Require group data to store all point data inpost
        if name in data.keys():
            del data[name]
        data.create_dataset(name, data=vector)

    add_postprocess_field = add_postprocess_data



class SimulationWriter(Simulation):

    # -------------------------------------------------------------------------
    # Writing methods
    # -------------------------------------------------------------------------

    def __init__(self, uid: str, path: str, comm: MPI.Comm = MPI.COMM_WORLD):
        super().__init__(uid, path, comm)
        self.step = 0

    def __enter__(self):
        self.change_status('Running')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val:
            self.change_status('Failed')
        else:
            self.change_status('Finished')


    def create(self) -> SimulationWriter:
        """Create a new file for this simlation."""
        self.step = 0
        self.global_fields = dict()
        if self._prank==0:
            if os.path.exists(self.h5file):
                os.remove(self.h5file)  # remove existing file if exists
        self.add_metadata()
        self.change_status('Initiated')

        return self

    def add_metadata(self) -> None:
        """Add metadata to h5 file."""
        nb_proc = self._comm.Get_size()
        if self._prank==0:
            with self._file('a'):
                self._file.attrs['time_stamp'] = str(datetime.datetime.now().replace(microsecond=0))
                self._file.attrs['id'] = self.uid
                self._file.attrs['processors'] = nb_proc
                self._file.attrs['notes'] = self._file.attrs.get('notes', "")

    def add_parameters(self, parameters: dict) -> None:
        """Add parameters to simulation.

        Args:
            parameters (dict): Dictionary with parameters.
        """
        if self._prank==0:
            with self._file('a'):
                # flatten parameters
                parameters = flatten_dict(parameters)

                if 'parameters' in self._file.keys():
                    del self._file['parameters']
                grp = self._file.create_group('/parameters')
                for key, val in parameters.items():
                    if isinstance(val, np.ndarray):
                        grp.create_dataset(key, data=val)
                    elif val is not None:
                        grp.attrs[key] = val
                    else:
                        pass

    def add_mesh(self, coordinates, connectivity, mesh_name: str = None) -> None:
        """Add the mesh to file. Currently only 2d meshes.

        Args:
            coordinates (np.array): Coordinates as array (nb_nodes, dim)
            connectivity (np.array): Connectivity matrix (nb_cells, nb nodes per cell)
            mesh_name (`str`): name for mesh (default = `base`)
        """
        if mesh_name is None:
            mesh_name = self._default_mesh
        # self._mesh_location = 'Mesh/0/mesh/'
        mesh_location = f'{self._mesh_location}/{mesh_name}/'
        
        nb_nodes_local = coordinates.shape[0]
        nb_cells_local = connectivity.shape[0]

        # gather total mesh
        nb_nodes_p = np.array(self._comm.allgather(nb_nodes_local))
        nb_cells_p = np.array(self._comm.allgather(nb_cells_local))
        nb_nodes, nb_cells = np.sum(nb_nodes_p), np.sum(nb_cells_p)

        # shape of datasets
        coord_shape = (nb_nodes, coordinates.shape[1]) if coordinates.ndim>1 else (nb_nodes,)
        conn_shape = (nb_cells, connectivity.shape[1]) if connectivity.ndim>1 else (nb_cells,)

        # global indices nodes
        idx_start = np.sum(nb_nodes_p[self._ranks<self._prank])
        idx_end = idx_start + nb_nodes_local

        # global indices cells
        idx_start_cells = np.sum(nb_cells_p[self._ranks<self._prank])
        idx_end_cells = idx_start_cells + nb_cells_local
        connectivity = connectivity + idx_start

        with self._file('a', driver='mpio', comm=self._comm) as f:
            if mesh_location in self._file.file_object:
                del self._file.file_object[mesh_location]
            grp = f.require_group(mesh_location)
            coord = grp.require_dataset('geometry', shape=coord_shape, dtype='f')
            conn  = grp.require_dataset('topology', shape=conn_shape, dtype='i')

            coord[idx_start:idx_end] = coordinates
            conn[idx_start_cells:idx_end_cells] = connectivity

            coord.flush()
            conn.flush()

    def add_field(self, name: str, vector: np.array,
                  time: float = None, mesh: str = None) -> None:
        """Add a dataset to the file. The data is stored at `data/`.

        Args:
            name (str): Name for the dataset
            vector (np.array): Dataset
            time (float): time
            mesh (str): Linked mesh for this data
        """
        if mesh is None:
            mesh = self._default_mesh

        # Get dimension of vector
        if vector.ndim<=1:
            vector = vector.reshape((-1, 1))
        dim = vector.shape[1]

        if time is None: time = self.step

        length_local = vector.shape[0]
        length_p = np.array(self._comm.allgather(length_local))
        length = np.sum(length_p)

        # global indices
        idx_start = np.sum(length_p[self._ranks<self._prank])
        idx_end = idx_start + length_local

        # open file
        with self._file('a', driver='mpio', comm=self._comm) as f:
            data = f.require_group('data')  # Require group data to store all point data in
            grp = data.require_group(name)
            vec = grp.require_dataset(str(self.step), shape=(length, dim), dtype='f')
            vec[idx_start:idx_end, :] = vector
            
            vec.attrs['t'] = time  # add time as attribute to dataset
            vec.attrs['mesh'] = mesh  # add link to mesh as attribute
            vec.flush()

    def add_global_field(self, name: str, value: float) -> None:
        """Add a gobal field. These are stored at `gloals/` as an array in a single dataset.

        Args:
            name (str): Name for the data
            value (float): Data
        """
        if self._prank==0:
            with self._file('a') as f:
                grp = f.require_group('globals')
                if name not in grp.keys():
                    vec = grp.create_dataset(name, shape=(1, ), dtype='f',
                                             chunks=True, maxshape=(None, ))
                    vec[0] = value
                else:
                    vec = grp[name]
                    vec.resize((self.step+1, ))
                    vec[-1] = value
                vec.flush()

    def add_additional(self, name: str, file: str) -> None:
        """Add an additional file stored elsewhere or in database directory. 

        Args:
            name: Name of data
            file: filename of file
        """
        if self._prank==0:
            with self._file('a') as f:
                grp = f.require_group('additionals')
                grp.attrs.update({name: file})

    def finish_step(self) -> None:
        """Finish step. Adds 1 to the step counter."""
        self.step += 1

    def finish_sim(self, status: str = 'Finished') -> None:
        if self._prank==0:
            self.change_status(status)

    def register_git_attributes(self, repo_path: str = './') -> None:
        """Register git information for given repo.

        Args:
            repo_path (`str`): path to git repository
        """
        if self._prank==0:
            repo_path = os.path.abspath(repo_path)
            # store current working directory
            cwd = os.getcwd()
            
            # switch directory to git repo
            os.chdir(repo_path)
            git_string = GitStateGetter().create_git_string()

            # switch working directory back
            os.chdir(cwd)

            with self._file('a') as f:
                grp = f.require_group('git')
                repo_name = os.path.split(repo_path)[1]
                print(f"Adding repo {repo_name}")
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

    def copy_file(self, path_to_file: str) -> None:
        """Copy a file to the datafolder.

        Args:
            path_to_file (`str`): path to file
        """
        shutil.copy(path_to_file, self.path)

