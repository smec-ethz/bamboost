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
import numpy as np
import pandas as pd
import h5py
import datetime
import logging
from types import SimpleNamespace
from mpi4py import MPI

from .xdmf import XDMFWriter
from .common.git_utility import GitStateGetter
from .common.job import Job
from .common.file_handler import open_h5file, with_file_open, H5Dataset
from .common.utilities import flatten_dict, unflatten_dict

log = logging.getLogger(__name__)



class Simulation:
    """A single dataset/simulation. Used to write to it, read from it or append.

    Args:
        uid (str): unique identifier
        path (str): path to parent/database folder
        comm (MPI.Comm): MPI communicator (default=MPI.COMM_WORLD)
    """
    opts = {
            'dataset.numpy': False,
            'dataset.h5py': False,
            }
    _file = None
    _mesh_location = 'Mesh/0/'

    def __init__(self, uid: str, path: str, comm: MPI.Comm = MPI.COMM_WORLD):
        self.uid = uid
        self.parent_path = path
        self.path = os.path.join(path, uid)
        self.h5file = os.path.join(self.path, f'{self.uid}.h5')
        self.xdmffile = os.path.join(self.path, f'{self.uid}.xdmf')
        os.makedirs(self.path, exist_ok=True)

        # MPI information
        self.comm = comm
        self.psize = self.comm.size
        self.prank = self.comm.rank
        self.ranks = np.array([i for i in range(self.psize)])

    @with_file_open('r')
    def __getitem__(self, path: str) -> h5py._hl.base:
        """Directly access data in the hdf5 file tree.

        Args:
            path (`str`): path inside hdf5 file
        Returns:
            h5py object at specified path
        """
        return self._file[path]

    def open(self, mode: str = 'r', driver=None, comm=None) -> h5py.File:
        """Open the HDF5 file.

        Args:
            mode (`str`): mode to open file (e.g. 'r')
        Returns:
            h5py file object
        """
        if self._file:
            if self._file.mode==mode:
                return self._file
            self.close()

        self._file = open_h5file(self.h5file, mode=mode, driver=driver, comm=comm)
        return self._file

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file:
            self._file.close()

    @property
    def parameters(self):
        tmp_dict = dict()
        if self.prank==0:
            with self.open('r') as file:
                tmp_dict.update(file['parameters'].attrs)
                for key in file['parameters'].keys():
                    tmp_dict.update({key: file[f'parameters/{key}'][()]})
        tmp_dict = unflatten_dict(tmp_dict)

        tmp_dict = self.comm.bcast(tmp_dict, root=0)
        return tmp_dict

    @property
    def metadata(self):
        tmp_dict = dict()
        if self.prank==0:
            with self.open('r') as file:
                tmp_dict.update(file.attrs)
        tmp_dict = self.comm.bcast(tmp_dict, root=0)
        return tmp_dict

    def update_metadata(self, update_dict: dict) -> None:
        """Update the metadata attributes.

        Args:
            update_dict: dictionary to push
        """
        with self.open('a') as file:
            file.attrs.update(update_dict)

    def update_parameters(self, update_dict: dict) -> None:
        """Update the parameters dictionary.

        Args:
            update_dict: dictionary to push
        """
        if self.prank==0:
            with self.open('a') as file:
                file['parameters'].attrs.update(update_dict)

    def create_xdmf_file(self, fields: list = None, nb_steps: int = None) -> None:
        """Create the xdmf file to read in paraview.

        Args:
            fields (list[str]): fields for which to write timeseries information,
                if not specified, all fields in data are written.
            nb_steps (int): number of steps the simulation has
        """

        if self.comm.rank==0:

            with self.open('r') as f:
                if not fields:
                    fields = list(f['data'].keys())

                if not nb_steps:
                    grp_name = list(f['data'].keys())[0]
                    nb_steps = list(f[f'data/{grp_name}'].keys())
                    nb_steps = max([int(step) for step in nb_steps])

            xdmf_writer = XDMFWriter(self.xdmffile, self.h5file)
            xdmf_writer.write_points_cells(f'{self._mesh_location}mesh/geometry',
                                           f'{self._mesh_location}mesh/topology')

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
                               f"--path {self.parent_path} --uid {self.uid}")
                    commands = [command]
            else:
                raise AttributeError("""Either you must specify an executable or have it 
                                     copied before with `copy_executable`!""")
        
        if euler:
            job.create_sbatch_script(commands, path=os.path.abspath(self.parent_path), uid=self.uid, nnodes=nnodes,
                                     ntasks=ntasks, ncpus=ncpus, time=time, mem_per_cpu=mem_per_cpu, tmp=tmp)
        else:
            job.create_bash_script_local(commands, path=os.path.abspath(self.parent_path),
                                         uid=self.uid, ntasks=ntasks)
        with self.open('a') as file:
            file.attrs.update({'submitted': False})

    def submit(self) -> None:
        """Submit the job for this simulation."""
        batch_script = os.path.abspath(os.path.join(self.path, f'sbatch_{self.uid}.sh'))
        subprocess.Popen(["sbatch", f"{batch_script}"])
        print(f'Simulation {self.uid} submitted!')

        with self.open('a') as file:
            file.attrs.update({'submitted': True})

    @with_file_open('a')
    def change_note(self, note) -> None:
        self._file.attrs['notes'] = note

    @with_file_open('a')
    def add_postprocess_field(self, name: str, vector: np.array) -> None:
        """Add a custom field in postprocessing."""
        data = self._file.require_group('postprocess')  # Require group data to store all point data in
        if name in data.keys():
            del data[name]
        data.create_dataset(name, data=vector)


class SimulationReader(Simulation):

    # -------------------------------------------------------------------------
    # Reading methods
    # -------------------------------------------------------------------------

    def print_hdf5_file_structure(self):
        """Print data in file."""
        with open_h5file(self.h5file, 'r') as file:
            def print_hdf5_item(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}")
            file.visititems(print_hdf5_item)

    @with_file_open('r')
    def access(self, path: str):
        """Simply access data in hdf5 file tree"""
        return self._dataset(self._file[f'{path}'])

    @property
    def mesh(self):
        """Return coordinates and connectivity of default mesh."""
        return self.get_mesh()

    @with_file_open('r')
    def get_mesh(self, mesh_name: str = 'mesh'):
        """Return coordinates and connectivity.

        Args:
            mesh_name (`str`): optional, name of mesh to read (default = mesh)
        """
        coordinates = self._dataset(self._file[f'{self._mesh_location}/{mesh_name}/geometry'])
        cells = self._dataset(self._file[f'{self._mesh_location}/{mesh_name}/topology'])
        return coordinates, cells

    @property
    @with_file_open('r')
    def globals(self):
        """Return globals."""
        grp = self._file['globals']
        d = {key: grp[key][()] for key in grp.keys()}
        return pd.DataFrame.from_dict(d)

    @property
    @with_file_open('r')
    def data_info(self) -> pd.Dataframe:
        """View the data stored."""
        tmp_dictionary = dict()
        for data in self._file['data'].keys():
            steps = max([int(step) for step in self._file[f'data/{data}'].keys()]) + 1
            shape = self._file[f'data/{data}/0'].shape
            dtype = self._file[f'data/{data}/0'].dtype
            tmp_dictionary[data] = {'dtype': dtype, 'shape': shape, 'steps': steps}
        return pd.DataFrame.from_dict(tmp_dictionary)

    @property
    @with_file_open('r')
    def git(self) -> dict:
        """Get Git information.

        Returns:
            Dictionary with different repositories.
        """
        if 'git' not in self._file.keys():
            return "Sorrrry, no git information stored :()"
        grp = self._file['git']
        tmp_dict = {}
        for repo in grp.keys():
            tmp_dict[repo] = grp[repo][()].decode('utf8')
        return tmp_dict

    @property
    def log(self) -> str:
        """Get exception traceback of failed simulations"""
        with open_h5file(self.h5file, 'r') as f:
            return f['log'][()].decode('utf8')

    @with_file_open('r')
    def data(self, name: str, step: int = None, time: float = None,
             read_linked_mesh: bool = False) -> SimpleNamespace:
        """Get the entire dataseries. Data can be extracted either at the specified
        step or at the specified time (nearest).

        Args:
            name (str): Name of dataset
            step (int): Step to extract
            time (float): Time to extract (closest available)
            read_linked_mesh (bool): If True, the mesh is returned too in the Namespace
        Returns:
            :class:`SimpleNamespace` with time(s) `t`, data `arr`, and mesh `mesh`.
        """
        grp = self._file[f'data/{name}']

        if read_linked_mesh:
            # Reads mesh of first step, needs thought if adaptive mesh is gonna be implemented
            linked_mesh = grp[str(0)].attrs.get('mesh', 'mesh')  # second mesh is default name
            mesh = self.get_mesh(linked_mesh)
        else:
            mesh = None

        # Return only specified step
        if step!=None:
            if step<0:
                last_step = max([int(i) for i in grp.keys()])
                step = last_step + (step + 1)
            time = grp[str(step)].attrs.get('t', step)
            return SimpleNamespace(t=time, arr=self._dataset(grp[str(step)]), mesh=mesh)

        # Return only specified time
        if time!=None:
            steps, times = [], []
            for step in grp.keys():
                times.append(grp[step].attrs['t'])
                steps.append(step)
            times = np.array(times)
            closest_step_idx = np.argmin(np.abs(times - time))
            return SimpleNamespace(t=times[closest_step_idx],
                                   arr=self._dataset(grp[steps[closest_step_idx]]),
                                   mesh=mesh)

        # Return full dataset if step not specified
        # Get times and number of steps
        times = list()
        step = 0
        while True:
            try:
                times.append(grp[str(step)].attrs.get('t', step))
                step += 1
            except KeyError:
                break

        try:
            vds = self._get_vds(name)
        except (KeyError, OSError):
            vds = self._create_vds(name, length=step)
        return SimpleNamespace(t=np.array(times), arr=vds, mesh=mesh)

    def _get_vds(self, name):
        """Access the virtual dataset."""
        vds_file = os.path.join(self.parent_path, '.database', '.virtual_datasets.h5')
        if not os.path.isfile(vds_file):
            raise OSError
        with open_h5file(vds_file, 'r') as f:
            vds = self._dataset(f[f'{self.uid}/{name}'], file=vds_file)
        return vds

    def _create_vds(self, name, length) -> None:
        """Create virtual dataset of the full timeseries of a field.
        Requires HDF5 > 1.10 !
        """
        vds_file = os.path.join(self.parent_path, '.database', '.virtual_datasets.h5')
        grp = self._file[f'data/{name}']
        shape_i = grp['0'].shape
        dtype_i = grp['0'].dtype
        layout = h5py.VirtualLayout(shape=(length, *shape_i), dtype=dtype_i)
        
        for step in range(length):
            vsource = h5py.VirtualSource(grp[str(step)])
            layout[step] = vsource

        # Add virtual dataset to VDS file
        with open_h5file(vds_file, 'a') as f:
            grp = f.require_group(self.uid)
            if name in grp.keys():
                del grp[name]
            vds = grp.create_virtual_dataset(name, layout)
            vds = self._dataset(vds, file=vds_file)

        return vds

    @with_file_open('r')
    def _das(self, name: str, step: int) -> np.ndarray:
        """Convenience shortcut for data at step."""
        grp = self._file[f'data/{name}']
        if step<0:
            last_step = max([int(i) for i in grp.keys()])
            step = last_step + (step + 1)
        return self._dataset(grp[str(step)])
        
    def delete_virtuals(self) -> None:
        vds_file = os.path.join(self.parent_path, '.database', '.virtual_datasets.h5')
        with open_h5file(vds_file, 'a') as f:
            if self.uid in f.keys():
                del f[self.uid]

    @property
    def post(self) -> SimpleNamespace:
        """Return the data stored in the postprocess category.

        Returns:
            SimpleNamespace with all data.
        """
        with self.open('r') as f:
            data = f['postprocess']
            d = {key: data[key][()] for key in data.keys()}
        return SimpleNamespace(**d)

    def _dataset(self, ds: h5py._hl.dataset, file: str = None) -> h5py._hl.dataset:
        if file==None: file = self.h5file
        if self.opts['dataset.h5py']:
            return ds
        ds = H5Dataset(file, ds.name)
        if self.opts['dataset.numpy']:
            return ds[()]
        return ds

    def get_data_interpolator(self, name: str, step: int):
        """Get Linear interpolator for data field.

        Args:
            name: name of the data field
            step: step
        """
        from scipy.interpolate import LinearNDInterpolator
        return LinearNDInterpolator(self.mesh[0], self.data(name, step).arr[()])


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
        if self.prank==0:
            if os.path.exists(self.h5file):
                os.remove(self.h5file)  # remove existing file if exists
        self.add_metadata()
        self.change_status('Initiated')

        return self

    def change_status(self, status: str) -> None:
        """Change status of simulation.

        Args:
            status (str): new status
        """
        if self.prank==0:
            with open_h5file(self.h5file, 'a') as f:
                f.attrs['status'] = status

    def add_metadata(self) -> None:
        """Add metadata to h5 file."""
        nb_proc = self.comm.Get_size()
        if self.prank==0:
            with open_h5file(self.h5file, 'a') as f:
                f.attrs['time_stamp'] = str(datetime.datetime.now().replace(second=0,
                                                                            microsecond=0))
                f.attrs['id'] = self.uid
                f.attrs['processors'] = nb_proc
                f.attrs['notes'] = f.attrs.get('notes', "")

    def add_parameters(self, parameters: dict) -> None:
        """Add parameters to simulation.

        Args:
            parameters (dict): Dictionary with parameters.
        """
        if self.prank==0:
            # flatten parameters
            parameters = flatten_dict(parameters)

            with open_h5file(self.h5file, 'a') as f:
                if 'parameters' in f.keys():
                    del f['parameters']
                grp = f.create_group('/parameters')
                for key, val in parameters.items():
                    if isinstance(val, np.ndarray):
                        grp.create_dataset(key, data=val)
                    elif val is not None:
                        grp.attrs[key] = val
                    else:
                        pass

    def add_mesh(self, coordinates, connectivity, mesh_name: str = 'mesh') -> None:
        """Add the mesh to file. Currently only 2d meshes.

        Args:
            coordinates (np.array): Coordinates as array (nb_nodes, dim)
            connectivity (np.array): Connectivity matrix (nb_cells, nb nodes per cell)
            mesh_name (`str`): name for mesh (default = `base`)
        """
        # self._mesh_location = 'Mesh/0/mesh/'
        mesh_location = f'Mesh/0/{mesh_name}/'
        
        nb_nodes_local = coordinates.shape[0]
        nb_cells_local = connectivity.shape[0]

        # gather total mesh
        nb_nodes_p = np.array(self.comm.allgather(nb_nodes_local))
        nb_cells_p = np.array(self.comm.allgather(nb_cells_local))
        nb_nodes, nb_cells = np.sum(nb_nodes_p), np.sum(nb_cells_p)

        # shape of datasets
        coord_shape = (nb_nodes, coordinates.shape[1]) if coordinates.ndim>1 else (nb_nodes,)
        conn_shape = (nb_cells, connectivity.shape[1]) if connectivity.ndim>1 else (nb_cells,)

        # global indices nodes
        idx_start = np.sum(nb_nodes_p[self.ranks<self.prank])
        idx_end = idx_start + nb_nodes_local

        # global indices cells
        idx_start_cells = np.sum(nb_cells_p[self.ranks<self.prank])
        idx_end_cells = idx_start_cells + nb_cells_local
        connectivity = connectivity + idx_start

        with open_h5file(self.h5file, 'a', driver='mpio', comm=self.comm) as f:
            grp = f.require_group(mesh_location)
            coord = grp.require_dataset('geometry', shape=coord_shape, dtype='f')
            conn  = grp.require_dataset('topology', shape=conn_shape, dtype='i')

            coord[idx_start:idx_end] = coordinates
            conn[idx_start_cells:idx_end_cells] = connectivity

            coord.flush()
            conn.flush()

    def add_field(self, name: str, vector: np.array,
                  time: float = None, mesh: str = 'mesh') -> None:
        """Add a dataset to the file. The data is stored at `data/`.

        Args:
            name (str): Name for the dataset
            vector (np.array): Dataset
            time (float): time
            mesh (str): Linked mesh for this data
        """
        # Get dimension of vector
        try:
            dim = vector.shape[1]
        except IndexError:  # then the vector is 1d/flat
            dim = 1
            vector = vector.reshape((-1, 1))

        if time is None: time = self.step

        length_local = vector.shape[0]
        length_p = np.array(self.comm.allgather(length_local))
        length = np.sum(length_p)

        # global indices
        idx_start = np.sum(length_p[self.ranks<self.prank])
        idx_end = idx_start + length_local

        # open file
        with open_h5file(self.h5file, 'a', driver='mpio', comm=self.comm) as f:
            data = f.require_group('data')  # Require group data to store all point data in
            grp = data.require_group(name)
            vec = grp.require_dataset(str(self.step), shape=(length, dim), dtype='f')
            vec[idx_start:idx_end, :] = vector
            if time:
                vec.attrs['t'] = time  # add time as attribute to dataset
                vec.attrs['mesh'] = mesh  # add link to mesh as attribute
            vec.flush()

    def add_global_field(self, name: str, value: float) -> None:
        """Add a gobal field. These are stored at `gloals/` as an array in a single dataset.

        Args:
            name (str): Name for the data
            value (float): Data
        """
        if self.prank==0:
            with open_h5file(self.h5file, 'a') as f:
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
        """Add an additional file. This is not necessary but helps to know what's around.

        Args:
            name: Name of data
            file: filename of file
        """
        if self.prank==0:
            with open_h5file(self.h5file, 'a') as f:
                grp = f.require_group('additionals')
                grp.attrs.update({name: file})

    def finish_step(self) -> None:
        """Finish step. Adds 1 to the step counter."""
        self.step += 1

    def finish_sim(self, status: str = 'Finished') -> None:
        if self.prank==0:
            self.change_status(status)

    def register_git_attributes(self, repo_path: str = './') -> None:
        """Register git information for given repo.

        Args:
            repo_path (`str`): path to git repository
        """
        if self.prank==0:
            repo_path = os.path.abspath(repo_path)
            # store current working directory
            cwd = os.getcwd()
            
            # switch directory to git repo
            os.chdir(repo_path)
            git_string = GitStateGetter().create_git_string()

            # switch working directory back
            os.chdir(cwd)

            with self.open('a') as f:
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

