# dbmanager

## Installation
Clone the repository, navigate into it and install it using pip:
```
pip install -e .
```
The option `-e` installs a project in editable mode from a local path. This way, you won't need to reinstall when pulling a new version or changing something in the package.

## Requirements

> `python 3.x` (if you're version is too low, it's very likely only because of typehints.
Please report and we can remove/change it)

`dbmanager` depends only on following packages:

- `numpy>?`
- `pandas`
- `h5py>?`
- `mpi4py`

## Usage
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

To be continued...
In the meantime, you can have a look at the example :)

## Feature requests / Issues
Please open issues on gitlab: [cmbm/dbmanager](https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager)
