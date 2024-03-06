Change log for bamboost
=======================

v0.5.1 (06Mar24)
--------------

- FIX: bugfix
- FIX: initialisation of `Simulation` does not create one anymore, if it doesn't
  exist.

v0.5.0 (05Mar24)
--------------

- FEAT: Made mpi4py an optional dependency (install `bamboost[mpi]` to include
  it). This allows to use bamboost without mpi4py, e.g. when installing
  `openmpi` is overkill.
- FEAT: Environment variable `BAMBOOST_NO_MPI` can be set to `1` to disable
  mpi4py even if it is installed.

If mpi4py is not used (or installed) a MockMPI class is used internally to
bypass any MPI code. Code should always still be written with MPI in mind.


v0.4.5 (13Feb24)
--------------

- API: sim.globals now returns a (hdf)Group object instead of a pandas dataframe
to be consistent with the other data accessors (sim.data, sim.userdata,
sim.meshes). The pandas dataframe is now accessible with `sim.globals.df`.
- API: Changed behaviour for context manager of `SimulationWriter`. Exiting does
  not change status to "Finished" anymore. Use `sim.finish_sim()` to do so.

v0.4.4 (12Feb24)
--------------

DOC: using typing_extension deprecated to tag deprecated functions
FEAT: added option `prefix` when creating a simulation (random id is prefixed with this string)
FEAT: added tiny script to bump the version (no CI publishing yet)
ENH: xdmf file is now printed with linebreaks for better readability
FIX: tiny cleanups

v0.4.3 (26Jan24)
--------------

#### SimulationWriter
API: data writer functions infer dtype for input. dtype can also be imposed


v0.4.2 (xxJan24)
--------------

FIX: Fixes for paralell writing. Writing is sped up significantly.
TEST: Added tests for paralell writing.
- Write 1000 steps with 1, 2, 4 and 8 threads. Timing is printed to stdout.
  `./tests/paralellization/test_steps/run.sh out_directory`
- Write single big array (20'000x20'000) with 1, 2, 4 and 8 threads. Timing is printed to stdout.
  `./tests/paralellization/test_big_array/run.sh out_directory`


v0.4.1
----------

Added functionality for `userdata`. To show the content of it, display the object `sim.userdata`.

### Simple way to store non field data
- Scalar, string, etc will are stored as attributes: `sim.userdata['some_name'] = value`
- Arrays are stored as datasets: `sim.userdata['array_name'] = arr`

Access is natural: `sim.userdata['some_name']` -> returns the stored value


v0.4.0
----------

Introduced globally unique identifiers for each database. They are stored as a file in the database directory. 
This allows to safely link different simulations together, e.g. meshes from a mesh database that are used in many places. Also, it allows to access any database from any path without remembering where it is stored. To do so, an index is maintained at `~.config/bamboost` which contains all previously accessed databases.

Also introduced nice reprs for a database and for a simulation inside jupyter notebooks :smile: 

In addition, some internal things were optimized (don't remember what exactly).
Hopefully, nothing was broken. Testing is still to do.

#### Simulation
Try the new nice repr in notebooks :)

**New methods:**
- `fromUID(full_uid: str)`: return simulation object from its full id
- `show_files()`: print the content of the simulation folder
- `show_h5tree()`: print the h5 file's structure
- `open_in_file_explorer()`: open the simulations folder in the default file explorer (linux only)
- `get_full_uid()`: returns the full id of the simulation (including the id of the database)

**New attributes:**
- `links`: Accessor to a mutable group in the h5 file to store and access linked simulations

#### Manager
Try the new nice repr in notebooks :)

**New attributes:**
- `fromUID`: access a database from anywhere by its UID (key completion shows known databases)
- `fromName`: access databases by name/path (key completion shows known databases)
- `FIX_DF`: new toggle. if set to false, the pandas dataframe will be reconstructed each time it is accessed.

#### Index
New module to manage the index of known databases, and finding them.
