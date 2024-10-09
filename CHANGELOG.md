# Change log for bamboost

## v0.8.0 (09Oct24)

- API (breaking): sim.mesh returns the default Mesh group instead of a tuple
  with coordinates and connectivity.
- API: add method add_fields to add a dictionary of key, array pairs to the
  simulation
- API: add method add_global_fields to add a dictionary of key, value pairs to
  the simulation
- API: create_simulation now takes additional kwargs: note, files (to copy), and
  links (to link)
- API: Group.extract_attrs() returns a dictionary of all attributes of all
  members of the group
- API: in mesh, renamed "geometry" to "coordinates" (consistent with Fenics).
  For backwards compatibility, "geometry" is still compatible.
- FEAT: added default argparse parsers in utility module
- ENH: split the add_field method to have access to a lower level _dump_array
  method which can be used to write data to any desired location in the hdf5
  file.
- ENH: create_simulation now deletes partially created simulations if an error
  occurs during creation
- ENH: some updates to the remote module
- CLEAN: remove deprecated method simulation_writer.add_additional
- FIX: duplicate check now works for multidimensional iterables
- FIX: fixed hdf5 file opening in XDMFWriter module

## v0.7.4 (03Sep24)

- FEAT: `Manager.sims` select argument now accepts a dictionary to filter for
  simulations. If select is a dictionary, `Manager.find` is called to find
  matches.

## v0.7.3 (26Aug24)

- API: sbatch jobs: removed default settings concerning MPI (sbatch will select the defaults).
  Starting from now, MPI related parameters (e.g. ntasts) have to be specified via `sbatch_kwargs` 
- FIX: fixed bug in manager that dataframe can't be displayed if one of the
  metadata keys is missing

## v0.7.2 (12Aug24)

- ENH: duplicate check speed up by using sql database
- ENH: duplicate check can compare lists and numpy arrays
- API: new method db.find() to search for matching simulations

## v0.7.1 (17Jul24)

- FEAT: Add possiblity to add custom sbatch options when creating sbatch
  submission script. E.g. "--mail=BEGIN,END,FAIL"
- FIX: numpy 2.0 compatibility
- FIX: boolean values are now correctly stored in the sqlite database
- FIX: When we try to open an h5 file, we now only put the program into a
  waiting state if we catch the `BlockedIOError`. All other `OSError` are raised
  immediately.
- FEAT: Improved logging. Bamboost logger available as
  `bamboost.BAMBOOST_LOGGER`. You can assign it a handler to log to a file, etc.
  See the logging module for more information.

## v0.7.0 (12Jun24)

**Major design change**: Using an sqlite database to store an up-to-date and cheaply
accessible "copy" of all bamboost databases, including parameters and metadata.
Designed as a caching system. The TRUE data remains the data in the hdf5 files.
Data remains self-describing even without any knowledge of bamboost. If the
sqlite database is lost/deleted corrupted, it is automatically rebuilt when
needed. Actuality is ensured by comparison of file modification times.
This also replaces the "unsafe" json database index we had before.

**Advantages:**

- Significant speed up for creation of metadata/parameter table (especially on
  Euler & for larger databases) because we avoid reading every single hdf5 file
  if it not necessary.
- Database IDs, Simulation IDs and metadata is available outside the bamboost
  ecosystem -> Useful for remote access (RemoteManager). We can now fetch the
  sqlite database and know of all our data and it's location on the remote.
- Speed up and availability of data very beneficial for the terminal user
  interface (tui) that is being developed in paralell.
- Database indexing is faster, less error-prone than before (json)

**Other news:**

- FEAT: Added CI tests (not extensive yet).
- FEAT: [Extension] Remote and RemoteManager classes to access databases on a remote server. Lazily caching requested simulations.
- FEAT: Config file at `~/.config/bamboost/config.toml` to store user settings
  (e.g. default sort order for tables, paths to search, etc.)
- FEAT: Command line interface to print index, table info, etc. to the terminal
  or to submit a simulation (e.g. all unsubmitted ones). Run `bamboostcli -h`
  for help.
- FEAT: Introduced TUI (terminal user interface). Not sure whether it should be included in
  this repo. (https://gitlab.com/zrlf/bamboost-tui)
- FEAT: utility function `show_differences` to show only the differences in
  pandas dataframes.

- API: requires `sqlite3 > 3.0` -> make sure to load the module sqlite
- API: paths to search moved to general config file `config.toml`

## v0.6.2 (02May24)

- FEAT: `Simulation`: Context manager that moves in and out of the simulation path
- FIX: Simulation: `submit` function fixed to work for paralell jobs

## v0.6.1 (26Apr24)

- API: `SimulationWriter.add_field` does not reshape 1D arrays to 2D anymore.
  Arrays are written as they are.
- FEAT: `SimulationWriter.add_field` now accepts arrays of any shape, not only
  1D or 2D arrays.
- FEAT: `SimulationWriter.add_field` now accepts a `center` argument to specify
  Cell or Node data.
- FEAT: new function `sim.open_paraview`
- FIX: fixed `XDMFWriter` to correctly write Tensors/Matrices and Cell/Node
  data.
- FIX: fixed `XDMFWriter` to correctly write Tensors/Matrices.

## v0.6.0 (25Apr24)

- FEAT: Added `extensions`.
  - `extensions.fenics.FenicsWriter`: optimized writer for FEniCS. Can write a fenics space directly.
  - `extensions.slurm.install()`: Monkey patches the simulation context
    manager, storing the slurm job info on Euler.
- FEAT: new utility function `show_differences` to show a DataFrame with only
  the differences between simulations.
- FIX: `sim.add_global_field` now works if previous steps were not written
- FIX: `add_mesh` dtype is now inferred from the input array

## v0.5.1 (06Mar24)

- FIX: bugfix
- FIX: initialisation of `Simulation` does not create one anymore, if it doesn't
  exist.

## v0.5.0 (05Mar24)

- FEAT: Made mpi4py an optional dependency (install `bamboost[mpi]` to include
  it). This allows to use bamboost without mpi4py, e.g. when installing
  `openmpi` is overkill.
- FEAT: Environment variable `BAMBOOST_NO_MPI` can be set to `1` to disable
  mpi4py even if it is installed.

If mpi4py is not used (or installed) a MockMPI class is used internally to
bypass any MPI code. Code should always still be written with MPI in mind.

## v0.4.5 (13Feb24)

- API: sim.globals now returns a (hdf)Group object instead of a pandas dataframe
  to be consistent with the other data accessors (sim.data, sim.userdata,
  sim.meshes). The pandas dataframe is now accessible with `sim.globals.df`.
- API: Changed behaviour for context manager of `SimulationWriter`. Exiting does
  not change status to "Finished" anymore. Use `sim.finish_sim()` to do so.

## v0.4.4 (12Feb24)

DOC: using typing_extension deprecated to tag deprecated functions
FEAT: added option `prefix` when creating a simulation (random id is prefixed with this string)
FEAT: added tiny script to bump the version (no CI publishing yet)
ENH: xdmf file is now printed with linebreaks for better readability
FIX: tiny cleanups

## v0.4.3 (26Jan24)

#### SimulationWriter

API: data writer functions infer dtype for input. dtype can also be imposed

## v0.4.2 (xxJan24)

FIX: Fixes for paralell writing. Writing is sped up significantly.
TEST: Added tests for paralell writing.

- Write 1000 steps with 1, 2, 4 and 8 threads. Timing is printed to stdout.
  `./tests/paralellization/test_steps/run.sh out_directory`
- Write single big array (20'000x20'000) with 1, 2, 4 and 8 threads. Timing is printed to stdout.
  `./tests/paralellization/test_big_array/run.sh out_directory`

## v0.4.1

Added functionality for `userdata`. To show the content of it, display the object `sim.userdata`.

### Simple way to store non field data

- Scalar, string, etc will are stored as attributes: `sim.userdata['some_name'] = value`
- Arrays are stored as datasets: `sim.userdata['array_name'] = arr`

Access is natural: `sim.userdata['some_name']` -> returns the stored value

## v0.4.0

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
