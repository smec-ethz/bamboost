# dbmanager

## Installation
Clone the repository, navigate into it and install it using pip:
```
pip install -e .
```
The option `-e` installs a project in editable mode from a local path. This way, you won't need to reinstall when pulling a new version or changing something in the package.

### h5py with parallel support
For mpi support, `h5py` must be installed with parallel support. Otherwise, each process writes one
after the other which takes forever. The default installation on Euler is not enough.

It's simple, do the following:
```
export CC=mpicc
export HDF5_MPI="ON"
pip install --force-reinstall --no-deps --no-binary=h5py h5py
```

## Requirements

> `python 3.x` (if you're version is too low, it's very likely only because of typehints.
Please report and we can remove/change it)

`dbmanager` depends only on following packages:

- `numpy>?`
- `pandas`
- `h5py>?`
- `mpi4py`

## Usage

### Manager
The dbmanager's main object is the `Manager`. It manages the database located in the directory
specified during construction. It can display all simulations, create new simulations, remove simulations
select a specific simulation based on it's `uid` or on conditions of it's parameters.
```python
from dbmanager import Manager

db = Manager(path)
```

The database can be viewed with:
```python
# interactive sessions (notebooks)
db.df

# scripts
print(db.df)
```

A simulation within a database can be viewed, retrieved and modified with the `Simulation` object.
There are various ways to get the `Simulation` object:
```python
sim = db.sim('uid')
sim = db.sim((db.df['parameter1']==2.0) & (db.df['time']>1))
sim = db[index]
```

All simulations can be returned as a (sorted) list:
```python
sims = db.sims()
sims = db.sims(sort='parameter1', reverse=False)
```


### Write data
You can use `dbmanagers` to write simulation or experimental data.
Use the `Manager` to create a new simulation (or access an existing one). Say you have (or want to create) a database at _data_path_.
The code sample below shows the main functionality. 

```python
from dbmanager import Manager

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
from dbmanager import SimulationWriter

with SimulationWriter(path, uid) as writer:

    # Do anything
```

### Read data
The key purpose is convenient access to data. I recommend an interactive session (notebooks).

#### Display database
```python
from dbmanager import Manager

db = Manager(data_path)
```

To display the database with its parametric space simply input
```python
db.df  # or also just `db`
```
Select a simulation of your dataset. `sim` will be a `SimulationReader` object.
```python
sim = db[index]
sim = db.sim(uid)
sim = db.sim((db.df.param1==2) & (db.df.param2>0), sort='param2')  # will return list of all matching, sorted by param2
```


Show data stored: This displays the stored fields and its sizes.
```python
sim.data_info
```
Access a mesh: Will return a tuple where [0] is the coordinates, [1] is the connectivity.
```python
coords, conn = sim.mesh  # default mesh
coords, conn = sim.get_mesh(mesh_name=...)
```
Access field data:
```python
full_field = sim.data('field_data_1', read_linked_mesh=True)
full_field.t  # an array with the times
full_field.arr  # an array with the data indexed (time, node, dimension)
full_field.mesh  # the corresponding mesh, only returned if option set to True

single_step = sim.data('field_data_1', -1)
single_step.t  # the time at the step
single_step.arr  # the data aat the step

single_time = sim.data('field_data_1', time=1e-3)
single_time.t, single_time.arr  # same as above
```
Access global data:
```python
sim.globals
kinetic_energy = sim.globals.kinetic_energy
```

By default, data is read into RAM. If you have large datasets you may switch the option `dataset.ram=False`. Then, data will be returned as an accessor (h5py object). When using this option, you must `open()` and `close()` yourself.
```python
sim.opts['dataset.ram'] = False
sim.open()
arr = sim.data('field_data_1').arr
print(arr[::10, :, 0])  # Only when you slice it, data will be copied into memory
sim.close()  # When you close it, the data will be inaccessible
```

### Job management
You can use `dbmanager` to create euler jobs, and to submit them.
```python
from dbmanager import Manager
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

sim.submit()  # submits the job using slurm
```


To be continued...
In the meantime, you can have a look at the example :)

## Feature requests / Issues
Please open issues on gitlab: [cmbm/dbmanager](https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager)
