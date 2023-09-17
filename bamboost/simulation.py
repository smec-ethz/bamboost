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
import subprocess
import numpy as np
import pandas as pd
import h5py
import logging
from mpi4py import MPI
from typing import Tuple

from .xdmf import XDMFWriter
from .common.job import Job
from .common.file_handler import FileHandler, with_file_open
from .common.utilities import flatten_dict, unflatten_dict
from .common import hdf_pointer
from .accessors.meshes import MeshGroup
from .accessors.fielddata import DataGroup

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

        self._file = FileHandler(self.h5file)

        # Initialize groups to meshes, data and userdata. Create groups.
        self.meshes = MeshGroup(self._file)
        self.data = DataGroup(self._file, self.meshes)
        self.userdata = hdf_pointer.MutableGroup(self._file, '/userdata')

    @with_file_open()
    def __getitem__(self, key) -> hdf_pointer.BasePointer:
        """Direct access to HDF5 file.

        Returns:
            :class:`~bamboost.common.file_handler.BasePointer`
        """
        new_pointer = hdf_pointer.get_best_pointer(self._file.file_object[key])
        return new_pointer(self._file, key)

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
            job.create_sbatch_script(commands, path=os.path.abspath(self.path_database),
                                     uid=self.uid, nnodes=nnodes, ntasks=ntasks,
                                     ncpus=ncpus, time=time, mem_per_cpu=mem_per_cpu, 
                                     tmp=tmp)
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


    # Ex-Simulation reader methods
    # ----------------------------

    def open(self, mode: str = 'r', driver=None, comm=None) -> FileHandler:
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
    def mesh(self):
        """Return coordinates and connectivity of default mesh.

        Returns:
            Tuple of np.arrays (coordinates, connectivity)
        """
        return self.get_mesh()

    @with_file_open('r')
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
    @with_file_open('r')
    def globals(self) -> pd.DataFrame:
        """Return global data.

        Returns:
            :class:`pd.DataFrame`
        """
        grp = self._file['globals']
        d = {key: grp[key][()] for key in grp.keys()}
        return pd.DataFrame.from_dict(d)

    @property
    @with_file_open('r')
    def data_info(self) -> pd.Dataframe:
        """View the data stored.

        Returns:
            :class:`pd.DataFrame`
        """
        tmp_dictionary = dict()
        for data in self.data:
            steps = len(data)
            shape = data.obj['0'].shape
            dtype = data.obj['0'].dtype
            tmp_dictionary[data._name] = {'dtype': dtype, 'shape': shape, 'steps': steps}
        return pd.DataFrame.from_dict(tmp_dictionary)

    @property
    @with_file_open('r')
    def git(self) -> dict:
        """Get Git information.

        Returns:
            :class:`dict` with different repositories.
        """
        if 'git' not in self._file.keys():
            return "Sorrrry, no git information stored :()"
        grp = self._file['git']
        tmp_dict = {}
        for repo in grp.keys():
            tmp_dict[repo] = grp[repo][()].decode('utf8')
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
        return LinearNDInterpolator(self.data[field].mesh.coordinates,
                                    self.data[field].at_step(step))

    @with_file_open()
    def print_hdf5_file_structure(self, print_datasets: bool = False):
        """Print data structure in file to screen."""
        def print_hdf5_item(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset) and print_datasets:
                print(f"Dataset: {name}")
        self._file.visititems(print_hdf5_item)



