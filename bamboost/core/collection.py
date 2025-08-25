"""Collection management module for bamboost.

This module provides the Collection class and related utilities for managing
collections of simulations in the bamboost framework. It includes functionality
for creating, filtering, querying, and manipulating simulation collections,
as well as integration with the underlying index and MPI communication.

Classes:
    Collection: Main interface for interacting with a simulation collection.
    _CollectionPicker: Helper for selecting collections by UID.
    _FilterKeys: Helper for key completion and filtering.

Functions:
    (See Collection methods for main API.)

"""

from __future__ import annotations

import pkgutil
from ctypes import ArgumentError
from functools import cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Mapping, Optional, cast

import numpy as np
import pandas as pd
from typing_extensions import deprecated

from bamboost import BAMBOOST_LOGGER, config
from bamboost._typing import StrPath
from bamboost.core.simulation.base import Simulation, SimulationName, SimulationWriter
from bamboost.core.utilities import flatten_dict
from bamboost.exceptions import DuplicateSimulationError, InvalidCollectionError
from bamboost.index import (
    CollectionUID,
    Index,
    create_identifier_file,
    get_identifier_filename,
)
from bamboost.index._filtering import Filter, Operator, _Key
from bamboost.index.sqlmodel import FilteredCollection
from bamboost.mpi import Communicator
from bamboost.mpi.utilities import RootProcessMeta
from bamboost.plugins import ElligibleForPlugin

if TYPE_CHECKING:
    from bamboost.index.sqlmodel import CollectionORM
    from bamboost.mpi import Comm

__all__ = [
    "Collection",
]

log = BAMBOOST_LOGGER.getChild("Collection")


class _CollectionPicker:
    def __getitem__(self, key: str, /) -> Collection:
        key = key.split(" - ", 1)[0]
        return Collection(uid=key)

    def _ipython_key_completions_(self):
        return (f"{i.uid} - {i.path[-30:]}" for i in Index.default.all_collections)


class _FilterKeys:
    def __init__(self, collection: Collection):
        self.collection = collection

    def __getitem__(self, key: str) -> _Key:
        return _Key(key)

    def _ipython_key_completions_(self):
        metadata_keys = (
            "collection_uid",
            "name",
            "created_at",
            "modified_at",
            "description",
            "status",
        )
        return (*self.collection._orm.get_parameter_keys()[0], *metadata_keys)


class Collection(ElligibleForPlugin):
    """
    Represents a collection of simulations in the bamboost framework.

    The Collection class provides an interface for managing, querying, and manipulating
    a group of simulations stored in a directory, with support for filtering, indexing,
    and MPI communication.

    Args:
        path: Path to the directory of the collection. If it doesn't exist,
            a new collection will be created if `create_if_not_exist` is True.
        uid: Unique identifier (UID) of the collection. If provided, the collection
            is resolved by UID instead of path.
        create_if_not_exist: If True (default), creates the collection directory if it does not exist.
        comm: MPI communicator to use for parallel operations. If not provided,
            the default communicator is used.
        index_instance: Custom Index instance to use for managing collections.
            If not provided, the default index is used.
        sync_collection: If True (default), synchronizes the collection with the index/database
            on initialization.
        filter: Optional filter to apply to the collection, returning a filtered view.

    Attributes:
        uid (CollectionUID): Unique identifier for the collection.
        path (Path): Filesystem path to the collection directory.
        df (pd.DataFrame): DataFrame view of the collection and its parameter space.
        k (_FilterKeys): Helper for key completion and filtering.
        FROZEN (bool): If True, the collection does not look for new simulations after initialization.

    Examples:
        >>> db = Collection("path/to/collection")
        >>> db.df  # DataFrame of the collection
        >>> sim = db["simulation_name"]  # Access a simulation by name
        >>> filtered = db.filter(db.k["param"] == 42)
    """

    FROZEN = False  # TODO: If true, the collection doesn't look for new simulations after initialization
    uid: CollectionUID
    path: Path
    fromUID = _CollectionPicker()
    _comm = Communicator()

    def __init__(
        self,
        path: Optional[StrPath] = None,
        *,
        uid: Optional[str] = None,
        create_if_not_exist: bool = True,
        comm: Optional[Comm] = None,
        index_instance: Optional[Index] = None,
        sync_collection: bool = True,
        filter: Optional[Filter] = None,
    ):
        assert path or uid, "Either path or uid must be provided."
        assert not (path and uid), "Only one of path or uid must be provided."

        self._index = index_instance or Index.default
        self._filter = filter

        # A key store with completion of all the parameters and metadata keys
        self.k = _FilterKeys(self)

        # Resolve the path (this updates the index if necessary)
        self.path = Path(path or self._index.resolve_path(uid.upper())).absolute()

        # Create the diretory for the collection if necessary
        if not self.path.is_dir():
            if not create_if_not_exist:
                raise NotADirectoryError("Specified path does not exist.")

            if self._comm.rank == 0:
                self.path.mkdir(parents=True, exist_ok=False)  # create the directory
                log.info(f"Initialized directory for collection at {path}")

        try:
            # Use uid if provided, otherwise resolve it from the path
            self.uid = CollectionUID(uid or self._index.resolve_uid(self.path))
        except InvalidCollectionError:
            # If the collection does not exist, create it in the index
            # and generate a new UID
            self.uid = CollectionUID()
            self._index.upsert_collection(self.uid, self.path)

        # Check if identifier file exists (and create it if necessary)
        if not self.path.joinpath(get_identifier_filename(uid=self.uid)).exists():
            create_identifier_file(self.path, self.uid)

        if sync_collection:
            # Sync the SQL table with the filesystem
            # Making sure the collection is up to date in the index
            self._index.sync_collection(self.uid, self.path)

            # Wait for root process to finish syncing
            self._comm.barrier()

    @property
    def _orm(self) -> CollectionORM | FilteredCollection:
        """
        Returns the ORM (Object Relational Mapping) object for the collection.

        If a filter is applied to the collection, returns a FilteredCollection
        object that represents the filtered view. Otherwise, returns the base
        CollectionORM object for the collection.

        Returns:
            CollectionORM or FilteredCollection: The ORM object representing the collection,
            possibly filtered.
        """
        collection_orm = self._index.collection(self.uid)
        if self._filter is None:
            return collection_orm
        return FilteredCollection(collection_orm, self._filter)

    def __len__(self) -> int:
        return len(self._orm.simulations)

    def __getitem__(self, name_or_index: str | int) -> Simulation:
        """
        Retrieve a Simulation from the collection by name or index.

        Args:
            name_or_index: The name of the simulation (str) or its index (int) in the collection dataframe.

        Returns:
            Simulation: The corresponding Simulation object.

        Raises:
            IndexError: If the index is out of range.
            KeyError: If the simulation name does not exist in the collection.

        Examples:
            >>> sim = collection["simulation_name"]
            >>> sim = collection[0]
        """
        if isinstance(name_or_index, int):
            name = self.df.iloc[name_or_index]["name"]
        else:
            name = name_or_index
        return Simulation(name, self.path, self._comm, collection_uid=self.uid)

    @cache
    def _ipython_key_completions_(self):
        return tuple(s.name for s in self._orm.simulations)

    def _repr_html_(self) -> str:
        """HTML repr for ipython/notebooks, using jinja2 for templating."""
        from jinja2 import Template

        html_string = pkgutil.get_data("bamboost", "_repr/manager.html").decode()
        icon = pkgutil.get_data("bamboost", "_repr/icon.txt").decode()
        template = Template(html_string)

        return template.render(
            icon=icon,
            db_path=f"<a href={self.path.as_posix()}>{self.path}</a>",
            db_uid=self.uid,
            db_size=len(self),
            filtered=self._filter is not None,
            filter=str(self._filter),
        )

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representing the collection and its parameter space.

        The DataFrame contains all simulations in the collection, including their parameters
        and metadata. The table is sorted according to the user-specified key and order
        in the configuration, if available.

        Returns:
            pd.DataFrame: DataFrame of the collection's simulations and parameters.
        """
        df = self._orm.to_pandas()

        # Try to sort the dataframe with the user specified key
        try:
            df.sort_values(
                config.options.sortTableKey,
                inplace=True,
                ascending=config.options.sortTableOrder == "asc",
                ignore_index=True,
            )
        except KeyError:
            pass

        return df

    def filter(self, *operators: Operator) -> Collection:
        """
        Returns a new Collection filtered by the given operators.

        This method applies the specified filter operators to the collection and returns
        a new Collection instance representing the filtered view. The original collection
        remains unchanged.

        Args:
            *operators: One or more filter operators (e.g., comparisons using Collection.k)
                to apply to the collection.

        Returns:
            Collection: A new Collection instance containing only the simulations that
            match the specified filter criteria.

        Examples:
            >>> filtered = collection.filter(collection.k["param"] == 42)
        """
        return Collection(
            path=self.path,
            create_if_not_exist=False,
            sync_collection=False,
            comm=self._comm,
            index_instance=self._index,
            filter=Filter(*operators) & self._filter,
        )

    def all_simulation_names(self) -> list[str]:
        """
        Returns a list of all simulation names in the collection.

        Returns:
            list[str]: A list containing the names of all simulations in the collection.
        """
        return [sim.name for sim in self._orm.simulations]

    def sync_cache(self, *, force_all: bool = False) -> None:
        """
        Synchronize the database for this collection.

        This method updates the collection's cache by syncing the underlying
        index and filesystem. It ensures that the collection's metadata and simulation
        information are up to date. If `force_all` is True, a full rescan and update
        of all simulations in the collection will be performed, regardless of their
        current cache state.

        Args:
            force_all: If True, force a full resync of all simulations in the collection.
                If False (default), only update simulations that are out of sync.
        """
        self._index.sync_collection(self.uid, self.path, force_all=force_all)

    def add(
        self,
        name: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        *,
        duplicate_action: Literal["ignore", "replace", "skip", "raise"] = "raise",
        description: Optional[str] = None,
        files: Optional[Iterable[str]] = None,
        links: Optional[Dict[str, str]] = None,
        override: bool = False,
    ) -> SimulationWriter:
        """
        Create and initialize a new simulation in the collection, returning a SimulationWriter object.

        This method is designed for parallel use, such as in batch scripts or parameter sweeps,
        where multiple simulations may be created concurrently. It handles creation of the simulation
        directory, duplicate checking, copying files, and setting up metadata and parameters.

        Args:
            name: The name/UID for the simulation. If not specified, a unique random ID
                will be generated.
            parameters: Dictionary of simulation parameters. If provided, these parameters
                will be checked against existing simulations for duplication. If not provided,
                parameters can be set later via `bamboost.core.simulation.Simulation.parameters`.

                Note:
                    - Parameters are stored in the HDF5 file as attributes.
                    - If a value is a dict, it is flattened using `bamboost.core.utilities.flatten_dict`.
                    - If a value is a list or array, it is stored as a dataset.

            duplicate_action: Action to take if a simulation with the same parameters already exists.
                Options are: "ignore" (create anyway), "replace" (delete existing and create new),
                "skip" (return existing simulation), "raise" (default, raise DuplicateSimulationError).
            description: Optional description for the simulation.
            files: Optional iterable of file paths to copy into the simulation directory.
                Each file will be copied with its original name.
            links: Optional dictionary of symbolic links to create in the simulation directory,
                mapping link names to target paths.
            override: If True, overwrite any existing simulation with the same name.
                If False (default), raises FileExistsError if a simulation with the same name exists.

        Returns:
            SimulationWriter: An object for writing data and metadata to the new simulation.

        Raises:
            FileExistsError: If a simulation with the same name already exists and override is False.
            ValueError, PermissionError: If there is an error during simulation creation.
            DuplicateSimulationError: If parameters are provided and a simulation with the same
                parameters already exists. Specify `duplicate_action` to control behavior.

        Examples:
            >>> db.add(parameters={"a": 1, "b": 2})

            >>> db.add(name="my_sim", parameters={"a": 1, "b": 2})

        Note:
            - The files and links specified are copied or created in the simulation directory.
            - This method is safe for use in parallel (MPI) environments.
        """
        import shutil

        assert duplicate_action in ("ignore", "replace", "skip", "raise"), (
            "Invalid duplicate_action. Must be one of: 'ignore', 'replace', 'skip', 'raise'."
        )

        name = SimulationName(name)  # Generates a unique id as name if not provided
        directory = self.path.joinpath(name)

        # Check if name is already in use, otherwise create a new directory
        if self._comm.rank == 0:
            exists = directory.exists()
            must_fail = exists and not override
            fail_msg = (
                f"Simulation {name} already exists in {self.path}" if must_fail else ""
            )
        else:
            exists = must_fail = None
            fail_msg = ""

        # Broadcast the decision to everyone
        must_fail = self._comm.bcast(must_fail, root=0)
        fail_msg = self._comm.bcast(fail_msg, root=0)

        if must_fail:
            raise FileExistsError(fail_msg)

        # Root process creates the directory if it does not exist
        if self._comm.rank == 0:
            # Check for duplicate parameters if provided
            if parameters and duplicate_action != "ignore" and not override:
                try:
                    self._check_duplicate(parameters)
                except DuplicateSimulationError as e:
                    if duplicate_action == "raise":
                        raise
                    elif duplicate_action == "skip":
                        log.info(
                            f"Simulation with parameters {parameters} already exists as"
                            f"{e.duplicates}. Skipping creation and returning first duplicate."
                        )
                        return self[e.duplicates[0]].edit()
                    elif duplicate_action == "replace":
                        log.info(
                            f"Removing (all) existing simulations with the given parameters: {e.duplicates}."
                        )
                        self.delete(e.duplicates)

            if exists and override:
                with RootProcessMeta.comm_self(self):
                    self._index._drop_simulation(self.uid, name)
                shutil.rmtree(directory)  # remove the old directory

            # finally create the new directory
            directory.mkdir(exist_ok=False)

        self._comm.barrier()

        try:
            # Create the simulation instance
            sim = SimulationWriter(
                name, self.path, self._comm, self._index, collection_uid=self.uid
            )
            with sim._file.open("w", driver="mpio"), self._index.sql_transaction():
                sim.initialize()  # create groups, set metadata and status
                sim.metadata["description"] = description or ""
                sim.parameters.update(parameters or {})
                sim.links.update(links or {})
                sim.copy_files(files or [])

            # Invalidate the file_map such that it is reloaded
            sim._file.file_map.invalidate()

            log.info(f"Created simulation {name} in {self.path}")
            return sim
        except (ValueError, PermissionError):
            log.error(
                f"Error occurred while creating simulation {name} at path {self.path}"
            )
            self._index._drop_simulation(self.uid, name)
            shutil.rmtree(directory)
            raise

    @deprecated(
        "Use `add` instead. This method has been renamed in v0.10.2 and will be"
        "removed in future versions."
    )
    def create_simulation(self, *args, **kwargs) -> SimulationWriter:
        """Deprecated alias of `add`. See `bamboost.core.collection.Collection.add` method."""
        return self.add(*args, **kwargs)

    def delete(self, name: str | Iterable[str]) -> None:
        """
        CAUTIOUS. Deletes one or more simulations from the collection.

        This method removes the specified simulation(s) from both the filesystem
        and the index/database. It is a destructive operation and should be used
        with caution.

        Args:
            name: Name of the simulation to delete, or an iterable of names.

        Raises:
            ValueError: If any of the specified names are invalid or do not exist in the collection.
            PermissionError: If there is an error deleting the simulation directory.

        Examples:
            >>> db.delete("simulation_name")
            >>> db.delete(["sim1", "sim2", "sim3"])
        """
        import shutil

        if isinstance(name, str):
            names = [name]
        elif isinstance(name, Iterable):
            names = list(name)
        else:
            raise ArgumentError("name must be a string or an iterable of strings.")

        # Check that all names exist in the collection
        existing_names = set(self.all_simulation_names())
        for n in names:
            if n not in existing_names:
                raise ValueError(f"Simulation {n} does not exist in the collection.")

        # Only root process performs deletion
        if self._comm.rank == 0:
            with RootProcessMeta.comm_self(self), self._index.sql_transaction():
                for n in names:
                    self._index._drop_simulation(self.uid, n)
                    try:
                        shutil.rmtree(self.path.joinpath(n))
                    except PermissionError as e:
                        log.error(f"Error deleting simulation directory for {n}: {e}")
                        raise

        self._comm.barrier()

    def find(self, parameter_selection: Mapping[str, Any]) -> pd.DataFrame:
        """
        Find simulations matching the given parameter selection.

        The parameter_selection dictionary can specify exact values for parameters,
        or use callables (such as lambda functions) for more complex filtering,
        such as inequalities or custom logic.

        Args:
            parameter_selection: Dictionary mapping parameter names to values
                or callables. If a value is a callable, it will be used as a filter function
                applied to the corresponding parameter column.

        Returns:
            pd.DataFrame: DataFrame containing simulations that match the specified criteria.

        Examples:
            >>> db.find({"a": 1, "b": lambda x: x > 2})
            >>> db.find({"a": 1, "b": 2})
        """
        parameter_selection = flatten_dict(parameter_selection)
        params = {}
        filters = {}
        for key, val in parameter_selection.items():
            if callable(val):
                filters[key] = val
            else:
                params[key] = val

        df = self.df
        matches = self._list_duplicates(params, df=df)
        matches = df[df.name.isin(matches)]
        assert isinstance(matches, pd.DataFrame)
        if len(matches) == 0:
            return matches

        for key, func in filters.items():
            matches = cast(pd.DataFrame, matches[matches[key].apply(func)])

        return matches

    def _list_duplicates(
        self, parameters: Mapping, *, df: pd.DataFrame | None = None
    ) -> list[str]:
        """
        List the names (IDs) of simulations in the collection that have duplicate parameter values.

        Args:
            parameters (dict): Parameter dictionary to check for duplicates. Keys are parameter names,
                values are the values to match against existing simulations.
            df (pd.DataFrame, optional): DataFrame to search in. If not provided, the
                DataFrame from the SQL database is used.

        Returns:
            list[str]: List of simulation names (IDs) that have the same parameter values as provided.
        """
        import pandas as pd

        if df is None:
            df = self._orm.to_pandas()
        params = flatten_dict(parameters)

        class ComparableIterable:
            def __init__(self, ori):
                self.ori = np.asarray(ori)

            def __eq__(self, other):
                return np.array_equal(np.asarray(other), self.ori)

        # make all iterables comparable by converting them to ComparableIterable
        for k in params.keys():
            if isinstance(params[k], Iterable) and not isinstance(params[k], str):
                params[k] = ComparableIterable(params[k])

        # if any of the parameters is not in the dataframe, no duplicates
        for p in params:
            if p not in df.keys():
                return []

        # get matching rows where all values of the series are equal to the corresponding values in the dataframe
        s = pd.Series(params)
        match = df.loc[(df[s.keys()].apply(lambda row: (s == row).all(), axis=1))]
        return match.name.tolist()

    def _check_duplicate(self, parameters: Mapping) -> Literal[True]:
        """
        Check whether the given parameters dictionary already exists in the collection.
        Returns True if no duplicates are found. Raises `DuplicateSimulationError` if
        duplicates are found.

        Args:
            parameters (dict): Parameter dictionary to check for duplicates.
        """

        duplicates = self._list_duplicates(parameters)
        if len(duplicates) == 0:
            return True

        raise DuplicateSimulationError(duplicates=tuple(duplicates))
