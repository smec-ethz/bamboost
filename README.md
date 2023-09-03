<div align="center">

<h3 align="center">
	<img src="./assets/bamboost_icon.png" width="150" alt="Logo"/><br/>
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


<!-- <div align="center"> -->
<!--     <img src="./assets/header_readme.svg" width="100%" alt="Header"/><br/> -->
<!-- </div> -->

## Installation
Clone the repository, navigate into it and install it using pip:
```
pip install -e .
```
The option `-e` installs a project in editable mode from a local path. This way,
you won't need to reinstall when pulling a new version or changing something in the
package.

### h5py with parallel support
For mpi support, `h5py` must be installed with parallel support. Otherwise, eachp
process writes one after the other which takes forever. The default installation on
Euler is not enough.

It's simple, do the following:
```
export CC=mpicc
export HDF5_MPI="ON"
pip install --force-reinstall --no-deps --no-binary=h5py h5py
```

## Requirements

> `python 3.x` (if you're version is too low, it's very likely only because of typehints.
Please report and we can remove/change it)

`bamboost` depends only on following packages:

- `numpy>?`
- `pandas`
- `h5py>?`
- `mpi4py`

## Usage

### Manager
The main object of `bamboost` is the `Manager`. It manages the database located in the directory
specified during construction. It can display the parametric space, create new simulations, remove simulations
select a specific simulation based on it's `uid` or on conditions of it's parameters.
```python
from bamboost import Manager

db = Manager(path)
```

`pandas` is used to display the database:
```python
db.df
```

A simulation within a database can be viewed, retrieved and modified with the `Simulation` object.
There are various ways to get the `Simulation` object:
```python
sim = db['uid']
sim = db[index]
sim = db.sim('uid')
```

All simulations can be returned as a (sorted) list. The argument `select` can be used to
filter the simulations:
```python
sims = db.sims()  # returns all
sims = db.sims(select=(db.df.eps==1))  # returns all where eps is 1
sims = db.sims(sort='parameter1', reverse=False)  # returns all, sorrted by parameter1
```

### Manage databases
Will be added in the future. All opened databases will be remembered, giving you 
easy access to your databases wherever they may be on the disk. Because it's likely our
brilliance makes us forget where we put our stuff.


### Write data
You can use `bamboost` to write simulation or experimental data.
Use the `Manager` to create a new simulation (or access an existing one).
Say you have (or want to create) a database at `data_path`.
The code sample below shows the main functionality. 

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

If you have an existing dataset. Do the following. You will need to pass the path and the uid to the script (best use `argparse`).
```python
from bamboost import SimulationWriter

with SimulationWriter(path, uid) as writer:

    # Do anything
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


**Show data stored:** This displays the stored fields and its sizes.
```python
sim.data_info
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
field1[:], field1[0, :]  # slice the dataset and you get numpy arrays
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
