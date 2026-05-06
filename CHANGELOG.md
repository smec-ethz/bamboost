# Change log for bamboost

## [0.13.5](https://github.com/smec-ethz/bamboost/compare/0.13.4...0.13.5) (2026-04-23)


### Bug Fixes

* **collection:** assign default for _include_links ([935cce7](https://github.com/smec-ethz/bamboost/commit/935cce715137e3ff5cc921cb3f9f5cc1945b876b))
* **filtering:** allow for lists in filtering operations ([#32](https://github.com/smec-ethz/bamboost/issues/32)) ([b648761](https://github.com/smec-ethz/bamboost/commit/b648761836f2b773b46e3259a455f7f7c03b1c3b))

## [0.13.4](https://github.com/smec-ethz/bamboost/compare/0.13.3...0.13.4) (2026-04-10)


### Bug Fixes

* **index:** branch in resolve path doesn't return resolved_uid if return_uid=True ([8c82ec6](https://github.com/smec-ethz/bamboost/commit/8c82ec6b319c93d5021bb3e37e6bf4f142689e42))

## [0.13.3](https://github.com/smec-ethz/bamboost/compare/0.13.2...0.13.3) (2026-04-09)


### Bug Fixes

* **collection:** uid does not default to alias if given uid is an alias ([464f838](https://github.com/smec-ethz/bamboost/commit/464f83865bfe2f164c4e3c876926bb88c4fb0bdd))
* **file:** drop eager creation of a full filemap when opening file ([89f82bc](https://github.com/smec-ethz/bamboost/commit/89f82bc227fb1daa7b7b43443a74d8719c186c89))
* **index:** collection is found from path if uid is alias ([1653052](https://github.com/smec-ethz/bamboost/commit/1653052f724a0b62c8ba1602bc23dab585f40b47))
* **index:** collections upsert statement returns list of inserts if ([e704392](https://github.com/smec-ethz/bamboost/commit/e7043921e3a14530f6abb70154d3f0153d8042c2))
* **index:** resolve collection path & uid if alias given ([e704392](https://github.com/smec-ethz/bamboost/commit/e7043921e3a14530f6abb70154d3f0153d8042c2))
* **simulation:** simplify status equality check ([7ee7a98](https://github.com/smec-ethz/bamboost/commit/7ee7a989cf9f35cfdeff9979d9fdecade24b731e))
* **simulation:** simplify status equality check ([1fc96b0](https://github.com/smec-ethz/bamboost/commit/1fc96b0b97af28f7901bacb7d4540b3d98ce0a0a))
* **test:** temp index now only contains the temp directory in searchPaths ([963c85e](https://github.com/smec-ethz/bamboost/commit/963c85ef87ee975eeed9417aca72f2d0156e5b33))


### Performance Improvements

* **file:** also add parent group to filemap when expand_group is called ([7f5e0c3](https://github.com/smec-ethz/bamboost/commit/7f5e0c35b81e4159ceab05dbad940cfaf6d223e4))

## [0.13.2](https://github.com/smec-ethz/bamboost/compare/0.13.1...0.13.2) (2026-03-27)


### Bug Fixes

* **collection:** ensure exact parameter matching for duplicate check ([2aa2e2e](https://github.com/smec-ethz/bamboost/commit/2aa2e2e0d9c5d247f15bc0084f48f95f3d96c457))
* **collection:** include links in duplicate check ([9007193](https://github.com/smec-ethz/bamboost/commit/90071933cd100f6159ee640ae81f47474b7ff0cb))
* **collection:** validate parameter keys to avoid conflicts ([8d7b5cf](https://github.com/smec-ethz/bamboost/commit/8d7b5cfbd3c4168b70f5ab3ece4387f3341bbdde))
* **index:** add links as json in simulation table ([fb1e2b5](https://github.com/smec-ethz/bamboost/commit/fb1e2b506ec49affdc526dc9d43ee933fd2f93fc))
* **index:** make check_integrity respect current search paths ([#14](https://github.com/smec-ethz/bamboost/issues/14)) ([7ab0254](https://github.com/smec-ethz/bamboost/commit/7ab0254d58009c428154cb448808653daa8b9f13))

## [0.13.1](https://github.com/smec-ethz/bamboost/compare/0.13.0...0.13.1) (2026-03-24)


### Bug Fixes

* **cli:** use relative index import to undo regression in startup time ([b3bcf78](https://github.com/smec-ethz/bamboost/commit/b3bcf7881ae6885e813fc83f8ceb02bb6406af5a))
* **config:** add options config.index.strictLinks/strictLinksWhenSyncing ([dba58ad](https://github.com/smec-ethz/bamboost/commit/dba58ade8b9324faa7ecade7723be0fb3ff38de6))
* **index:** soften enforcement of existing targets of links ([dba58ad](https://github.com/smec-ethz/bamboost/commit/dba58ade8b9324faa7ecade7723be0fb3ff38de6))

## [0.13.0](https://github.com/smec-ethz/bamboost/compare/0.12.1...0.13.0) (2026-03-23)


### Features

* **collection:** add `include_links()` to bundle link parameters in df ([421ca74](https://github.com/smec-ethz/bamboost/commit/421ca74e57d68a518dc8c7c369adcbf7ab728236))
* **index:** add helper functions to get links and backlinks ([9512594](https://github.com/smec-ethz/bamboost/commit/95125943178ef5bd45f3716ae1e7effc4497f140))


### Bug Fixes

* **config:** add `options.logLevel` config option ([1c2d64f](https://github.com/smec-ethz/bamboost/commit/1c2d64fdfac7c9fe069989d16d00fde5181dcd2f))
* **hdf5:** support attributes/parameters of type None ([5c9185c](https://github.com/smec-ethz/bamboost/commit/5c9185c3d6a78efc85d3ccf16fdfea4a2b5f7425))
* **index:** fix inserting links when starting with an empty database & other minor fixes ([254ac8e](https://github.com/smec-ethz/bamboost/commit/254ac8e0cda8a19e2ddbce39ef61999d856f7e88))
* **remote:** local db name would append version repeateadly ([39ba824](https://github.com/smec-ethz/bamboost/commit/39ba82499a4548928022abde603c46a0b8f6ab9b))
* **remote:** remote base path is added twice when rsyncning index ([#9](https://github.com/smec-ethz/bamboost/issues/9)) ([270d77a](https://github.com/smec-ethz/bamboost/commit/270d77abc6a3867e177d521e16db7c8ae822899e))


### Performance Improvements

* **index:** include links in sqlite database in separate links table ([9e484e3](https://github.com/smec-ethz/bamboost/commit/9e484e3cc4418458f95bde6b3ec357be4cbb15e4))
* **index:** vectorize filtering of collections ([d26ab4d](https://github.com/smec-ethz/bamboost/commit/d26ab4d0ce054f87d0d603e40d55ae4b5a0c3f3c))

## [0.12.1](https://github.com/smec-ethz/bamboost/compare/0.12.0...0.12.1) (2026-02-17)


### Bug Fixes

* **cli:** add `bamboost` console entry point ([fd431e8](https://github.com/smec-ethz/bamboost/commit/fd431e869a37710b269a245e645678558e7f1d68))

## [0.12.0](https://github.com/smec-ethz/bamboost/compare/0.11.3...0.12.0) (2026-02-11)


### Features

* **collection:** add easy filtering based on tags ([b77ba86](https://github.com/smec-ethz/bamboost/commit/b77ba861956d96af37ed52ac59b3e72096cfe83a))
* **simulation:** add tags for simulations ([234eb25](https://github.com/smec-ethz/bamboost/commit/234eb254e379c86128d7ab7abf4f22139218231f))


### Bug Fixes

* **index:** add schema drift handling (pragma columns, add any missing column) ([6771095](https://github.com/smec-ethz/bamboost/commit/6771095bbdc20d24d0f09103751582194dfe4fe3))

## 0.11.3 (2025-12-02)

### Fix

- **mpi**: raise runtime error if h5py has no mpi but bamboost wants it (8ade683) #57

## 0.11.2 (2025-12-01)

### Fix

- **index**: (fixes #54) assert collection is stored in database if uid is unknown and found from path (20cde7f)

## 0.11.1 (2025-11-25)

### Fix

- **remote**: eagerly rsync remote simulation if directory does not exist locally (10f0e1f)
- **remote**: allow fetching of different version (old db) and auto migrate (57398d0)
- **collection**: respect sort order in the dataframe (8a2a1a4)
- **remote**: fix naming of local sqlite databases (2615e41)
- **mpi**: fix uncorrect wrapping of staticmethod and classmethod when using RootProcessMeta meta class (7e841ae)
- **remote**: bug fixes in remote collection (e32fcb4)
- **remote**: filtering a remote collection now returns a remote collection (3da8b60)
- **collection**: type annotation for files in Collection.add (c744b7c)

### Refactor

- **collection**: add collection.\_replace similar to NamedTuple or dataclass shallow copying (311577b)
- **index**: introduce extensible schema versioning used for migration between them (db1fba2)

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

- **hdf5**: fix **str** (print) of attrsdict instances, e.g. parameters (494fd1e)
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
- **simulation**: assign \_file attribute in constructor instead of a cached_property (fixes a type issue)

## 0.10.0 (2025-06-04)

### Fix

- **sql**: cascade delete of parameters
- **coll**: fixes for simulation generation under mpi
- **coll**: make creation of collections and simulations possible when run with mpi

### Refactor

- **hdf5**: introduce ArrayLike protocol to use instead of hardcoded numpy array (to allow for jax arrays)

## 0.10.0a0 (2025-05-09)

Start of a new change log starting from 0.10.0 release series.
