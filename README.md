<div align="center">

<h3 align="center">
	<img src="https://gitlab.com/cmbm-ethz/bamboost/-/raw/main/assets/bamboost_icon.png?ref_type=heads" width="150" alt="Logo"/><br/>
        <br/>
	BAMBOOST <br/>

</h3>

<p align="center">
Bamboost is a Python library built for datamanagement using
the HDF5 file format.
bamboost stands for a <span style="font-weight: bold;">lightweight</span> shelf which will <span style="font-weight: bold">boost</span> your efficiency and which
will totally break if you load it heavily. Just kidding, bamboo can fully carry pandas. <br/>
🐼🐼🐼🐼
</p>

</div>


<div align="center">
<a href="https://bamboost.ch">
    <img src="https://img.shields.io/badge/Docs-bamboost.ch-blue" alt="Website">
</a>
<!-- <a href="https://gitlab.com/cmbm-ethz/bamboost/-/commits/main"> -->
<!--     <img src="https://gitlab.com/cmbm-ethz/bamboost/badges/main/pipeline.svg" alt="pipeline status"> -->
<!-- </a> -->
<!-- <a href="https://gitlab.com/cmbm-ethz/bamboost/-/commits/main"> -->
<!--     <img src="https://gitlab.com/cmbm-ethz/bamboost/badges/main/coverage.svg" alt="coverage report"> -->
<!-- </a> -->
<a href="https://badge.fury.io/py/bamboost">
    <img src="https://badge.fury.io/py/bamboost.svg" alt="PyPI version">
</a>
<a href="https://pypistats.org/packages/bamboost">
    <img src="https://img.shields.io/pypi/dm/bamboost" alt="PyPI - Downloads">
</a>
<a href="https://pypi.org/project/bamboost/">
    <img src="https://img.shields.io/pypi/pyversions/bamboost" alt="PyPI - Python Version">
</a>
<a href="https://gitlab.com/cmbm-ethz/bamboost/-/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/bamboost" alt="PyPI - License">
</a>

<a href="https://gitlab.com/cmbm-ethz/bamboost/-/commits/test-ci"><img alt="pipeline status" src="https://gitlab.com/cmbm-ethz/bamboost/badges/test-ci/pipeline.svg" /></a> 
<a href="https://gitlab.com/cmbm-ethz/bamboost/-/commits/test-ci"><img alt="coverage report" src="https://gitlab.com/cmbm-ethz/bamboost/badges/test-ci/coverage.svg" /></a>

Terminal user interface for bamboost: https://gitlab.com/zrlf/bamboost-tui.

Documentation site: https://gitlab.com/zrlf/bamboost-docs.
</div>

## Installation
Install the latest release from the package repository:
```
pip install bamboost
```


Install the package in editable mode for more flexibility, $e.g.$ if you plan to make changes yourself:
```
git clone git@gitlab.com:cmbm-ethz/bamboost.git
cd bamboost
pip install -e .
```
> :warning: The option `-e` installs a project in editable mode from a local path. This way,
> you won't need to reinstall when pulling a new version or changing something in the
> package.

### MPI support
For parallel writing, you need `mpi4py`. Furthermore, the HDF5 library must have
been built with MPI support and the python interface `h5py` must be installed
with parallel support (https://docs.h5py.org/en/stable/mpi.html).
```
export CC=mpicc
export HDF5_MPI="ON"
pip install --force-reinstall --no-deps --no-binary=h5py h5py
```

> :warning: If you're system runs into problems installing `mpi4py`, make sure python header files are installed. Quickly google what you need (something like `python3-dev`, `libpython3.8-dev`, etc.). 


## Requirements

- **Python** >= 3.7 
- **HDF5** >= 1.10.0
- **sqlite3** (comes with python) >= 3.0

Optional:
- **MPICC** or **OpenMPI** for parallel writing
- **HDF5** with MPI support


    
## Usage

### Manager
The main object of `bamboost` is the `Manager`. It manages the database located in the directory
specified during construction. It can display the parametric space, create new simulations, remove simulations
select a specific simulation based on it's `uid` or on conditions of it's parameters.
Every database that is created is assigned a unique identifier (UID).
```python
from bamboost import Manager

db = Manager('path/to/db')
```

`pandas.DataFrame` is used to display the database. The dataframe is convenient and fast to filter or sort your entries:
```python
db.df
```

An entry (from now on called *simulation*) within a database can be viewed, retrieved and modified with the `Simulation` object.
To get the `Simulation` object, access it with it's identifier or location (index) in the dataframe:
```python
sim = db['uid']
sim = db[index]
sim = db.sim('uid')
```

All simulations can be returned as a (sorted) list. The argument `select` can be used to
filter the simulations. 
```python
sims = db.sims()  # returns all
sims = db.sims(select=(db.df.eps==1))  # returns all where eps is 1
sims = db.sims(sort='parameter1', reverse=False)  # returns all, sorrted by parameter1
```
:warning: Note that this creates objects for every simulation and the sorting is not optimized. Using pandas to select and sort is much faster. Check their documentation for how to manipulate pandas dataframes.

### Database index
Every database created will be assigned a unique identifier (UID). 
The database path is stored with the UID in an index maintained at `~.config/bamboost` in your home directory. If it is not known, `bamboost` will try to find it on your disk (you can add paths to search in `~.config/bamboost/known_paths.json`).
You can obtain a Manager object of any database from anywhere with it's UID. In notebooks, key completion will show you all known databases:
```python
db = Manager.fromUID['UID']
``` 

The unique id makes refering to data safe. The full identifier of a simulation is considered to be `'(database id):(simulation id)'`. It is encouraged to use the identifiers (instead of the path) to link from one simulation to a different one.
```python
# add a link to a different simulation (e.g. the mesh)
sim.links['mesh_to_use'] = 'DATABASE-ID:simulation-id'

# the full id of a simulation is accessible as such
uid = sim.get_full_uid()
```


### Write data
You can use `bamboost` to write simulation or experimental data.
Use the `Manager` to create a new simulation (or access an existing one).
Say you have (or want to create) a database at `data_path`.
The code samples below shows the main functionality. 

```python
from bamboost import Manager

db = Manager(data_path)

params = {...}  # dictionary of parameters (can have nested dictionaries)
writer = db.create_simulation(parameters=params)
writer.copy_file('path/to/file/')  # copy a file which is needed to the database folder (e.g. executable, module list, etc.)
writer.change_note('This run is part of a series in which i investigate the effect of worms on apples')

# Use context manager (with block) and the file will be tagged 'running', 'finished', 'failed' automatically
with writer:

    writer.add_metadata()  # adds time and other metadata
    writer.add_mesh(coordinates, connectivity)  # Add a mesh, default mesh is named 'mesh'. 
    writer.add_mesh(coordinates, connectivity, mesh_name='interface')  # add a second mesh for e.g. the interface
    
    # loop through your time data and write
    for t in times:
        writer.add_field('field_data_1', array, time=t)
        writer.add_field('field_data_2', array, time=t, mesh='interface')
        writer.add_global_field('kinetic_energy', some_number)
        writer.finish_step()  # this increases the step counter
```

If you have an existing dataset, $e.g.$ because you created the simulation before and it holds the input parameters or similar. Do the following: You will need to pass the path and the uid to the script (best use `argparse`).
```python
from bamboost import SimulationWriter

with SimulationWriter(path, uid) as writer:

    # Do anything
```

#### Userdata (Data not related to time and/or space)
The above functionality should be used for ordered data, such as timeseries of spatial data related to a mesh.
For anything else, there is the `userdata` category. You can use it to store (almost) anything in the simulation file structured how you would like it. This is also useful to store computed values during postprocessing or plotting.
Internally, Userdata is an object handling a specific group ('/userdata') of the hdf5 file. To show the content of the group, display the object:
```python
sim.userdata
```

 You can create a subgroup, which will return a self-similar object for the new group ($e.g.$ '/userdata/plots'):
```python
plot_grp = sim.userdata.require_group('plots')
```

Writing something to the file (group) is as easy as:
```python
sim.userdata['avg_T'] = 34.56256
sim.userdata['traction_profile'] = np.array([...])
```

And reading:
```python
# read avg_T
sim.userdata['avg_T']
# read dataset traction_profile
sim.userdata['traction_profile']  
# note that this returns an object Dataset. To actually read the array, you will need to slice it
sim.userdata['traction_profile'][:]
```


### Read data
The key purpose is convenient access to data. I recommend an interactive session (notebooks).

#### Display database
```python
from bamboost import Manager

db = Manager(data_path)
```

To display the database with its parametric space simply input
```python
db.df
```
Select a simulation of your dataset. `sim` will be a `SimulationReader` object.
```python
sim = db[index]
sim = db[uid]
sims = db.sims((db.df.param1==2) & (db.df.param2>0), sort='param2')  # will return list of all matching, sorted by param2
```


**Show data stored:** 
Display content of the data, userdata, globals groups:
```
sim.data
sim.userdata
sim.globals
```

This displays the stored fields and its sizes.
```python
sim.data.info
```
**Access a mesh:** Directly access a tuple where [0] is the coordinates, [1] is the connectivity.
```python
coords, conn = sim.mesh  # default mesh
coords, conn = sim.get_mesh(mesh_name=...)
```
You can get a mesh object the following way.
```python
mesh1 = sim.meshes['mesh1']
mesh1.coordinates  # gives coordinates
mesh1.connectivity  # gives connectivity
mesh1.get_tuple()  # gives both the above
```

**Access field data**:
`sim.data` acts as an accessor for all field data.
```python
field1 = sim.data['field1']
field1[:], field1[0, :]  # slice the dataset and you get numpy arrays (time, *spatial)
field1.at_step(-1)  # similar for access of one step
field1.mesh  # returns the linked mesh object (see above)
field1.msh  # returns a tuple of the mesh (coordinates, connectivity)
field1.coordinates, field1.connectivity  # direct access to linked mesh' coords and conn arrays
field1.times  # returns timesteps of data
field1.shape  # shape of data
field1.dtype  # data type of data
```

**Access global data:**
```python
sim.globals
kinetic_energy = sim.globals.kinetic_energy
```

**Open file:**
All methods internally open the HDF5 file and make sure that it is closed again. Sometimes
it's useful to keep the file open (i.e. to directly change something in the file manually).
To do so, you are encouraged to use the following.

> :warning: Do not open the file in write mode (`'w'`) as this truncates the file.
```python
with sim.open(mode='r+') as file:
    # do anything
    # in here, you can still use all functions of the bamboost, the functions will not close
    # the file in the case you manually opened the file...
```

### Job management
You can use `bamboost` to create euler jobs, and to submit them.
```python
from bamboost import Manager
db = Manager(data_path)
params = {...}  # dictionary of parameters (can have nested dictionaries)

sim = db.create_simulation(parameters=params)
sim.copy_file('path/to/postprocess.py')  # copy a file which is needed to the database folder (e.g. executable, module list, etc.)
sim.copy_file('path/to/cpp_script')
sim.change_note('This run is part of a series in which i investigate the effect of worms on apples')

# commands to execute in batch job
commands = []
commands.append('./cpp_script')
commands.append(f'mpirun python {os.path.join(sim.path, 'postprocess.py')}') # e.g. to write the output to the database from cpp output

sim.create_batch_script(commands, ntasks=4, time=..., mem_per_cpu=..., euler=True)

sim.submit()  # submits the job using slurm (works only in jupyterhub sessions on Euler)
```


To be continued...

## Feature requests / Issues
Please open issues on gitlab: [cmbm/bamboost](https://gitlab.ethz.ch/compmechmat/research/libs/bamboost)
