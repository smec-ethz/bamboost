# Change log for bamboost

## 0.11.0 (2025-10-17)

### Feat

- **cli**: add a migrate function to create the new 0.11 database from old one (c28b455)
- **collection**: add & and | operators to filtering (cf136c5)
- **collection**: explicitly define the iteration behaviour (6b820b4)
- **collection**: add sort functionality (e3eed58)
- **collection**: add support for any metadata fields (1bd8148)
- **cli**: add "yank" to copy uid, "config show/print" to open config file, "alias get/add/remove" to manage aliases (e7a1489)
- **config**: add clipboardCommand option to set the platform specific clipboard command (f11fd17)
- **cli**: add command to add and remove collection aliases (+ improvements) (071f1bc)
- **cli**: add drop command to drop collections from the database (6e18935)
- **collection**: add collection metadata and persistent storage in yaml (7adb633)

### Fix

- **hdf5**: fix __str__ (print) of attrsdict instances, e.g. parameters (494fd1e)
- **cli**: metadata save will update both the yaml and sqlite files (7305c29)
- **collection**: filter logic updated (a337b9a)
- **collection**: make metadata store mpi safe (0241fe0)
- **collection**: raise error if metafile is weird, instead of overwriting with new content (5679df7)
- **config**: eagerly convert project dir to path object (cc4a6b1)
- **cli**: fix display when syncing fs + add aliases in ls (f618dcd)
- **remote**: fix destination in rsync of remote collection (eeb54bf)
- **remote**: fix path to remote database (7f7156d)

### Refactor

- **collection**: refactor metadata handling (+add author field) (017b2a8)
- **config**: rename config file to config.toml, rename sqlite file to bamboost.sqlite (ded7308)
- **index**: rename sqlmodel.py -> store.py (566be79)
- **index**: fix two minor mistakes (7653324)
- **index**: completely replace ORM with a sqlalchemy core flow (d52facd)
- **index**: add methods to load and parse collection metadata (8f2aba4)
- **cli**: minor improvement to cli (add --entries argument to ls to limit print lines) (a4de3f3)

## 0.10.3 (2025-10-08)

### Fix

- **config**: ensure that the local directory to store the database exists (9535306)

## 0.10.2 (2025-08-25)

### Feat

- **collection**: add implementation of duplicate check to `coll.create_simulation` (newly renamed to `coll.add`)

### Fix

- **collection**: update Operand typealias to include datetime & timedelta in filtering

### Refactor

- **collection**: rename `create_simulation` to `add` for simplicity
- **collection**: bring delete function up to date
- replace wildcard imports with explicit re-exports

## 0.10.1 (2025-06-16)

### Feat

- **collection**: extend getitem such that coll[idx] returns the simulation for the provided index in df

### Fix

- **config**: store remainder of config file (additional config, e.g. for tui)

### Refactor

- **index**: raise InvalidCollectionError if calling resolve_uid with a path that is no collection (#51)
- **simulation**: assign _file attribute in constructor instead of a cached_property (fixes a type issue)

## 0.10.0 (2025-06-04)

### Fix

- **sql**: cascade delete of parameters
- **coll**: fixes for simulation generation under mpi
- **coll**: make creation of collections and simulations possible when run with mpi

### Refactor

- **hdf5**: introduce ArrayLike protocol to use instead of hardcoded numpy array (to allow for jax arrays)

## 0.10.0a0 (2025-05-09)

Start of a new change log starting from 0.10.0 release series.
