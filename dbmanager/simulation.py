from __future__ import annotations
import os
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import pandas as pd
import h5py
import datetime
from mpi4py import MPI
from functools import wraps


from .xdmf import XDMFWriter
from .common.git_utility import GitStateGetter
from .common.job import Job
from .common.file_handler import open_h5file
from .common.utilities import flatten_dict, unflatten_dict


def temporary_open_file(mode: str = 'r'):
    """Decorator.
    If opts['dataset.ram'] = False, the file should be opened and closed at end of function
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self.opts['dataset.ram']:
                with self.open(mode):
                    return method(self, *args, **kwargs)
            else:
                assert self._file, "If `dataset.ram` is OFF you need to manually open/close the file"
                if self._file.mode!=mode:
                    self.close()
                    self.open(mode)
                return method(self, *args, **kwargs)
        return wrapper
    return decorator


class Simulation:
    """A single dataset/simulation. Used to write to it, read from it or append.

    Args:
        uid (str): unique identifier
        path (str): path to parent/database folder
        comm (MPI.Comm): MPI communicator (default=MPI.COMM_WORLD)
    """
    opts = {
            'dataset.ram': True,
            }
    _file = None

    def __init__(self, uid: str, path: str, comm: MPI.Comm = MPI.COMM_WORLD):
        self.uid = uid
        self.parent_path = path
        self.path = os.path.join(path, uid)
        os.makedirs(self.path, exist_ok=True)
        self.h5file = os.path.join(self.path, f'{self.uid}.h5')
        self.xdmffile = os.path.join(self.path, f'{self.uid}.xdmf')
        self.mesh_location = 'Mesh/0/mesh/'

        # MPI information
        self.comm = comm
        self.psize = self.comm.size
        self.prank = self.comm.rank
        self.ranks = np.array([i for i in range(self.psize)])

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
            with open_h5file(self.h5file, 'r') as file:
                tmp_dict.update(file.attrs)
        tmp_dict = self.comm.bcast(tmp_dict, root=0)
        return tmp_dict

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
        if not nb_steps:
            if self.comm.rank==0:
                with self.open('r') as f:
                    grp_name = list(f['data'].keys())[0]
                    nb_steps = list(f[f'data/{grp_name}'].keys())
                    nb_steps = max([int(step) for step in nb_steps])

        if self.comm.rank==0:

            if not fields:
                with self.open('r') as f:
                    fields = list(f['data'].keys())

            xdmf_writer = XDMFWriter(self.xdmffile, self.h5file)
            xdmf_writer.write_points_cells(f'{self.mesh_location}geometry',
                                           f'{self.mesh_location}topology')

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
                    command = (f"{{MPI}} python3 {os.path.join(self.path, self.executable)}"
                               f"--path {self.path} --uid {self.uid}")
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

    @temporary_open_file('a')
    def change_note(self, note) -> None:
        self._file.attrs['notes'] = note

    @temporary_open_file('a')
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

    memmap = False

    def print_hdf5_file_structure(self):
        """Print data in file."""
        with open_h5file(self.h5file, 'r') as file:
            def print_hdf5_item(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}")
            file.visititems(print_hdf5_item)

    @temporary_open_file('r')
    def access(self, path: str):
        """Simply access data in hdf5 file tree"""
        return self._dataset(self._file[f'{path}'])

    @property
    @temporary_open_file('r')
    def mesh(self):
        """Return coordinates and connectivity."""
        coordinates = self._dataset(self._file[f'{self.mesh_location}/geometry'])
        cells = self._dataset(self._file[f'{self.mesh_location}/topology'])
        return coordinates, cells

    @property
    @temporary_open_file('r')
    def globals(self):
        """Return globals."""
        grp = self._file['globals']
        d = {key: grp[key][()] for key in grp.keys()}
        return pd.DataFrame.from_dict(d)

    @property
    @temporary_open_file('r')
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
    @temporary_open_file('r')
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

    @temporary_open_file('r')
    def data(self, name: str, step: int = None, time: float = None) -> SimpleNamespace:
        """Get the entire dataseries. Data can be extracted either at the specified
        step or at the specified time (nearest).

        Args:
            name (str): Name of dataset
            step (int): Step to extract
            time (float): Time to extract (closest available)
        Returns:
            :class:`SimpleNamespace` with time(s) `t` and data `arr`.
        """
        grp = self._file[f'data/{name}']

        # Return only specified step
        if step!=None:
            if step<0:
                last_step = max([int(i) for i in grp.keys()])
                step = last_step + (step + 1)
                time = grp[str(step)].attrs.get('t', step)
            return SimpleNamespace(t=time, arr=self._dataset(grp[str(step)]))

        # Return only specified time
        if time!=None:
            steps, times = [], []
            for step in grp.keys():
                times.append(grp[step].attrs['t'])
                steps.append(step)
            times = np.array(times)
            closest_step_idx = np.argmin(np.abs(times - time))
            return SimpleNamespace(t=times[closest_step_idx],
                                   arr=self._dataset(grp[steps[closest_step_idx]]))

        times = list()    
        step = 0
        while True:
            try:
                times.append(grp[str(step)].attrs.get('t', step))
                step += 1
            except KeyError:
                break

        # Return full dataset if step not specified
        if '_VDS' not in self._file.keys() or name not in self._file['_VDS'].keys():
            self._create_vds(name, step)

        return SimpleNamespace(t=np.array(times),
                               arr=self._dataset(self._file[f'_VDS/{name}']))

    def _dataset(self, ds: h5py._hl.dataset) -> h5py._hl.dataset:
        if self.opts['dataset.ram']:
            return ds[()]
        return ds

    @temporary_open_file('r+')
    def _create_vds(self, name, length) -> None:
        """Create virtual dataset of the full timeseries of a field.
        Requires HDF5 > 1.10 !
        """
        grp = self._file[f'data/{name}']
        shape_i = grp['0'].shape
        dtype_i = grp['0'].dtype
        layout = h5py.VirtualLayout(shape=(length, *shape_i), dtype=dtype_i)
        
        for step in range(length):
            vsource = h5py.VirtualSource(grp[str(step)])
            layout[step] = vsource

        # Add virtual dataset to VDS group
        self._file.require_group('_VDS')
        self._file.create_virtual_dataset(f'_VDS/{name}', layout)

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

    def add_mesh(self, coordinates, connectivity) -> None:
        """Add the mesh to file. Currently only 2d meshes.

        Args:
            coordinates (np.array): Coordinates as array (nb_nodes, dim)
            connectivity (np.array): Connectivity matrix (nb_cells, nb nodes per cell)
        """
        self.mesh_location = 'Mesh/0/mesh/'
        
        nb_nodes_local = coordinates.shape[0]
        nb_cells_local = connectivity.shape[0]

        # gather total mesh
        nb_nodes_p = np.array(self.comm.allgather(nb_nodes_local))
        nb_cells_p = np.array(self.comm.allgather(nb_cells_local))
        nb_nodes, nb_cells = np.sum(nb_nodes_p), np.sum(nb_cells_p)

        # global indices nodes
        idx_start = np.sum(nb_nodes_p[self.ranks<self.prank])
        idx_end = idx_start + nb_nodes_local

        # global indices cells
        idx_start_cells = np.sum(nb_cells_p[self.ranks<self.prank])
        idx_end_cells = idx_start_cells + nb_cells_local
        connectivity = connectivity + idx_start

        with open_h5file(self.h5file, 'a', driver='mpio', comm=self.comm) as f:
            grp = f.require_group(self.mesh_location)
            coord = grp.require_dataset('geometry', shape=(nb_nodes, 2), dtype='f')
            conn  = grp.require_dataset('topology', shape=(nb_cells, 3), dtype='i')

            coord[idx_start:idx_end] = coordinates
            conn[idx_start_cells:idx_end_cells] = connectivity

            coord.flush()
            conn.flush()

    def add_field(self, name: str, vector: np.array, time: float = None) -> None:
        """Add a dataset to the file. The data is stored at `data/`.

        Args:
            name (str): Name for the dataset
            vector (np.array): Dataset
            time (float): time
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


