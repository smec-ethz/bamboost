# Change log for bamboost

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
