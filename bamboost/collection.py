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
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
    Literal,
    Mapping,
    Optional,
    cast,
)

import pandas as pd
from bamboostrs import Collection as _Collection
from bamboostrs import get_collection
from typing_extensions import Self

from bamboost._config import config
from bamboost._logger import BAMBOOST_LOGGER
from bamboost._typing import StrPath
from bamboost.core.utilities import flatten_dict
from bamboost.filtering import Filter, Operator, Sorter, SortInstruction, _Key
from bamboost.mpi import Communicator, ReuseComm
from bamboost.mpi.utilities import parallel_proxy
from bamboost.simulation.base import Simulation, SimulationWriter
from bamboost.utilities import ComparableIterable

if TYPE_CHECKING:
    from bamboost.mpi import Comm

__all__ = [
    "Collection",
]

log = BAMBOOST_LOGGER.getChild("Collection")


class Collection:
    """Represents a collection of simulations in the bamboost framework.

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

    Examples:
        >>> db = Collection("path/to/collection")
        >>> db.df  # DataFrame of the collection
        >>> sim = db["simulation_name"]  # Access a simulation by name
        >>> filtered = db.filter(db.k["param"] == 42)
    """

    uid: str
    """Unique identifier of the collection."""
    path: Path
    """Path to the collection directory."""
    """Helper for selecting collections by UID."""
    _comm = Communicator()
    _filter: Filter | None = None
    """Internal variable to keep track of the filter applied to the collection. If None,
    no filter is applied."""
    _sorter: Sorter | None = None
    """Internal variable to keep track of the sort instructions applied to the collection.
    If None, no sorting is applied."""
    _include_links: Iterable[str] | Literal[True] | None = None
    """Internal variable to keep track of which links to include in the collection view.
    If True, includes all links."""

    _core: _Collection
    """Internal variable to hold the underlying core babo Collection."""

    def __init__(
        self,
        uid_or_path: StrPath,
        *,
        create_if_not_exist: bool = True,
        comm: Optional[Comm] = None,
        filter: Optional[Filter] = None,
        sorter: Optional[Sorter] = None,
        include_links: Iterable[str] | Literal[True] | None = None,
    ):
        if comm is not None:
            self._comm = comm

        if create_if_not_exist:
            self._core = parallel_proxy(get_collection, self._comm, 0, str(uid_or_path))
        else:
            self._core = parallel_proxy(_Collection, self._comm, 0, str(uid_or_path))

        self.uid = self._core.uid
        self.path = Path(self._core.path)

    def __len__(self) -> int:
        return len(self._core.parameter_space())

    def __getitem__(self, name_or_index: str | int) -> Simulation:
        """Retrieve a Simulation from the collection by name or index.

        Args:
            name_or_index: The name of the simulation (str) or its index (int) in the
                collection dataframe.

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
        return Simulation(name, self.path, ReuseComm(self))

    def __iter__(self) -> Generator[Simulation, None, None]:
        """Iterate over all simulations in the collection."""
        for sim in self._core.get_simulations():
            yield Simulation.from_core(sim)

    def _repr_html_(self) -> str:
        """HTML repr for ipython/notebooks, using jinja2 for templating."""
        from jinja2 import Template

        html_string = pkgutil.get_data("bamboost", "_repr/manager.html").decode()  # type: ignore
        icon = pkgutil.get_data("bamboost", "_repr/icon.txt").decode()  # type: ignore
        template = Template(html_string)

        return template.render(
            icon=icon,
            db_path=f"<a href={self.path.as_posix()}>{self.path}</a>",
            db_uid=self.uid,
            db_size=len(self),
            _filter=self._filter,
            _sort=self._sorter,
        )

    def _replace(self, **changes) -> Self:
        """Return a shallow copy of this Collection with some attrs replaced."""
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.__dict__.update(changes)
        return new

    def include_links(self, *keys: str) -> Self:
        """Returns a new Collection that includes parameters of simulations linked with
        the specified keys.

        Args:
            *keys: One or more keys representing the links to include. If no keys are
                provided, all linked simulations will be included.

        Returns:
            Collection: A new Collection instance that includes the linked simulations.

        Examples:
            >>> linked_collection = collection.include_links("key1", "key2")
            >>> all_linked = collection.include_links()  # include all linked simulations
        """
        if not keys:
            return self._replace(_include_links=True)

        return self._replace(_include_links=keys)

    def to_pandas(
        self, flatten: bool = True, include_links: bool = True
    ) -> pd.DataFrame:
        """Returns a pandas DataFrame representing the collection and its parameter space.

        The DataFrame contains all simulations in the collection, including their
        parameters and metadata. The table is sorted according to the user-specified key
        and order in the configuration, if available.

        Also includes parameters of linked simulations if `include_links` is set. (use
        `coll.include_links(...).df` to include specific links)

        Args:
            flatten: If True (default), flatten nested parameter dictionaries into a single
                level using dot notation. If False, keep nested dictionaries as they are.
            include_links: If True (default), the link (name and target) are included as
                columns.

        Returns:
            DataFrame of the collection's simulations and parameters.
        """
        df = pd.DataFrame(self._core.parameter_space())

        # apply filtering if necessary
        if self._filter is not None:
            df = cast(pd.DataFrame, self._filter.apply(df))

        # Try to sort the dataframe with the user specified key
        try:
            if self._sorter is None:
                df.sort_values(
                    config.options.sortTableKey,
                    inplace=True,
                    ascending=config.options.sortTableOrder == "asc",
                    ignore_index=True,
                )
        except KeyError:
            pass

        return df

    @property
    def df(self) -> pd.DataFrame:
        """Returns a pandas DataFrame representing the collection and its parameter space.

        The DataFrame contains all simulations in the collection, including their
        parameters and metadata. The table is sorted according to the user-specified key
        and order in the configuration, if available.

        Also includes parameters of linked simulations if `include_links` is set. (use
        `coll.include_links(...).df` to include specific links)

        Returns:
            pd.DataFrame: DataFrame of the collection's simulations and parameters.
        """
        return self.to_pandas()

    def filter(
        self, *operators: Operator, tags: str | Iterable[str] | None = None
    ) -> Self:
        """Returns a new Collection filtered by the given operators.

        This method applies the specified filter operators to the collection and returns a
        new Collection instance representing the filtered view. The original collection
        remains unchanged.

        Args:
            *operators: One or more filter operators (e.g., comparisons using Collection.k)
                to apply to the collection.
            tags: Optional tag or iterable of tags to filter by.

        Returns:
            Collection: A new Collection instance containing only the simulations that
            match the specified filter criteria.

        Examples:
            >>> filtered = collection.filter(collection.k["param"] == 42)
        """
        tags = (tags,) if isinstance(tags, str) else tags  # handle single string case
        return self._replace(_filter=Filter(*operators, tags=tags) & self._filter)

    def sort(self, key: _Key | str, ascending: bool = True) -> Self:
        """Returns a new Collection sorted by the given instructions.

        This method applies the specified sort instructions to the collection and returns
        a new Collection instance representing the sorted view. The original collection
        remains unchanged.

        Args:
            key: A SortInstruction object or a string representing the parameter or
                metadata key to sort by.
            ascending: If True (default), sorts in ascending order. If False, sorts in
                descending order.

        Returns:
            Collection: A new Collection instance with simulations sorted according to
            the specified instructions.

        Examples:
            >>> sorted_collection = collection.sort(SortInstruction("param", ascending=False))
        """
        if isinstance(key, _Key):
            key = key._value

        if self._sorter is None:
            new_sorter = Sorter(SortInstruction(key, ascending))
        else:
            new_sorter = self._sorter & Sorter(SortInstruction(key, ascending))

        return self._replace(_sorter=new_sorter)

    def all_simulation_names(self) -> list[str]:
        """Returns a list of all simulation names in the collection.

        Returns:
            list[str]: A list containing the names of all simulations in the collection.
        """
        return self._core.get_simulation_names()

    def sync_cache(self) -> None:
        """Synchronize the database for this collection.

        This method updates the collection's cache by syncing the underlying index and
        filesystem. It ensures that the collection's metadata and simulation information
        are up to date. If `force_all` is True, a full rescan and update of all
        simulations in the collection will be performed, regardless of their current cache
        state.

        Args:
            force_all: If True, force a full resync of all simulations in the collection.
                If False (default), only update simulations that are out of sync.
        """
        self._core.sync_cache()

    def add(
        self,
        name: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        *,
        duplicate_action: Literal["ignore", "replace", "skip", "raise"] = "raise",
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        files: Optional[Iterable[StrPath]] = None,
        links: Optional[dict[str, str]] = None,
        override: bool = False,
    ) -> SimulationWriter:
        """Create and initialize a new simulation in the collection, returning a
        SimulationWriter object.

        This method is designed for parallel use, such as in batch scripts or parameter
        sweeps, where multiple simulations may be created concurrently. It handles
        creation of the simulation directory, duplicate checking, copying files, and
        setting up metadata and parameters.

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
            tags: Optional sequence of tags for the simulation metadata.
            files: Optional iterable of file paths to copy into the simulation directory.
                Each file will be copied with its original name.
            links: Optional dictionary of symbolic links to create in the simulation
                directory, mapping link names to target paths.
            override: If True, overwrite any existing simulation with the same name. If
                False (default), raises FileExistsError if a simulation with the same name
                exists.

        Returns:
            SimulationWriter: An object for writing data and metadata to the new
            simulation.

        Raises:
            FileExistsError: If a simulation with the same name already exists and override is False.
            ValueError, PermissionError: If there is an error during simulation creation.
            DuplicateSimulationError: If parameters are provided and a simulation with the same
                parameters already exists. Specify `duplicate_action` to control behavior.

        Examples:
            >>> db.add(parameters={"a": 1, "b": 2})

            >>> db.add(name="my_sim", parameters={"a": 1, "b": 2})

        Note:
            - This method is safe for use in parallel (MPI) environments.
            - Be cautious when using `duplicate_action="replace"` as it will delete
              existing simulations with matching parameters, without asking again.
        """
        assert duplicate_action in ("ignore", "replace", "skip", "raise"), (
            "Invalid duplicate_action. Must be one of: 'ignore', 'replace', 'skip', 'raise'."
        )
        sim_core = self._core.add_simulation(
            name,
            dict(parameters) if parameters else None,
            source_files=[str(f) for f in files] if files else None,
            description=description,
            tags=list(tags) if tags else None,
            links=links,
            duplicate_action=duplicate_action,
        )
        return SimulationWriter.from_core(sim_core)

    def delete(self, name: str | Iterable[str]) -> None:
        """CAUTION. Deletes one or more simulations from the collection.

        This method removes the specified simulation(s) from both the filesystem and the
        index/database. It is a destructive operation and should be used with caution.

        Args:
            name: Name of the simulation to delete, or an iterable of names.

        Raises:
            ValueError: If any of the specified names are invalid or do not exist in the
                collection.
            PermissionError: If there is an error deleting the simulation directory.

        Examples:
            >>> db.delete("simulation_name")
            >>> db.delete(["sim1", "sim2", "sim3"])
        """

        if isinstance(name, str):
            names = [name]
        elif isinstance(name, Iterable):
            names = list(name)
        else:
            raise ArgumentError("name must be a string or an iterable of strings.")

        self._core.drop_simulations(names)

    def find(self, parameter_selection: Mapping[str, Any]) -> pd.DataFrame:
        """Find simulations matching the given parameter selection.

        The parameter_selection dictionary can specify exact values for parameters, or use
        callables (such as lambda functions) for more complex filtering, such as
        inequalities or custom logic.

        Args:
            parameter_selection: Dictionary mapping parameter names to values or
                callables. If a value is a callable, it will be used as a filter function
                applied to the corresponding parameter column.

        Returns:
            pd.DataFrame: DataFrame containing simulations that match the specified
            criteria.

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
        matches = self._match_parameters(params, df=df)
        matches = df[df.name.isin(matches)]
        assert isinstance(matches, pd.DataFrame)
        if len(matches) == 0:
            return matches

        for key, func in filters.items():
            matches = cast(pd.DataFrame, matches[matches[key].apply(func)])

        return matches

    def _match_parameters(
        self,
        parameters: Mapping | None = None,
        *,
        links: Mapping | None = None,
        df: pd.DataFrame | None = None,
        exact: bool = False,
    ) -> list[str]:
        """List the names (IDs) of simulations in the collection that have duplicate
        parameter values.

        Args:
            parameters: Parameter dictionary to check for duplicates. Keys are parameter
                names, values are the values to match against existing simulations.
            links: Dictionary of simulation links to include in the duplicate check.
            df: DataFrame to search in. If not provided, the DataFrame from the SQL
                database is used.
            exact: If True, only matches simulations that have exactly the same set of
                parameters. If False (default), matches simulations that have at least
                the provided parameters matching.

        Returns:
            list[str]: List of simulation names (IDs) that have the same parameter values
            as provided.
        """
        import pandas as pd

        if df is None:
            df = self.to_pandas(include_links=True)
        params = flatten_dict(parameters or {})
        if links:
            # Prefix links to match flattened DataFrame columns
            params.update(flatten_dict({"links": links}))

        # make all iterables comparable by converting them to ComparableIterable
        for k in params:
            if isinstance(params[k], Iterable) and not isinstance(params[k], str):
                params[k] = ComparableIterable(params[k])

        # if any of the parameters/links is not in the dataframe, no duplicates
        for p in params:
            if p not in df.columns:
                return []

        # get matching rows where all values of the series are equal to the corresponding values in the dataframe
        s = pd.Series(params)
        if s.empty:
            mask = pd.Series(True, index=df.index)
        else:
            # Treat missing values as equal when both sides are missing.
            # Pandas compares None/NaN as unequal with `==`, so combine equality
            # with a per-cell both-missing condition.
            subset = df[s.keys()]
            eq = subset.eq(s)
            both_missing = subset.isna() & s.isna()
            mask = (eq | both_missing).all(axis=1)

        if exact:
            # For exact match, all other parameter/link columns must be NaN/missing.
            # We exclude tags, description, and other internal metadata.
            metadata_cols = {
                "name",
                "created_at",
                "description",
                "tags",
                "status",
                "submitted",
            }
            other_cols = [
                c
                for c in df.columns
                if c not in s.keys() and c not in metadata_cols  # noqa: SIM118
            ]
            for col in other_cols:
                mask &= df[col].isna()

        return df.loc[mask].name.tolist()
