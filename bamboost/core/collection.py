from __future__ import annotations

import pkgutil
from ctypes import ArgumentError
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, cast

import numpy as np

from bamboost import BAMBOOST_LOGGER, config
from bamboost._typing import StrPath
from bamboost.core.simulation.base import Simulation, SimulationName, SimulationWriter
from bamboost.core.utilities import flatten_dict
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
    import pandas as pd

    from bamboost.index.sqlmodel import CollectionORM
    from bamboost.mpi import Comm

__all__ = [
    "Collection",
    "NotACollectionError",
]

log = BAMBOOST_LOGGER.getChild("Collection")


class NotACollectionError(NotADirectoryError):
    """Raised when a path is not a valid collection."""

    def __init__(self, path: Path):
        super().__init__(f"{path} is not a valid collection.")


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
    """View of database.

    Args:
        path: path to the directory of the database. If doesn't exist,
            a new database will be created.
        comm: MPI communicator
        uid: UID of the database

    Attributes:
        ...

    Example:
        >>> db = Manager("path/to/db")
        >>> db.df # DataFrame of the database
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

        # Resolve or get an UID for the collection (updates the index if necessary)
        self.uid = CollectionUID(uid or self._index.resolve_uid(self.path))

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
        collection_orm = self._index.collection(self.uid)
        if self._filter is None:
            return collection_orm
        return FilteredCollection(collection_orm, self._filter)

    def __len__(self) -> int:
        return len(self._orm.simulations)

    def __getitem__(self, name: str) -> Simulation:
        return Simulation(name, self.path, self._comm, collection_uid=self.uid)

    @cache
    def _ipython_key_completions_(self):
        return tuple(s.name for s in self._orm.simulations)

    def _repr_html_(self) -> str:
        """HTML repr for ipython/notebooks. Uses string replacement to fill the
        template code.
        """
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
        """View of the collection and its parametric space.

        Returns:
            A dataframe of the collection
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
        """Filter the collection with the given operators.

        Args:
            *operators: The operators to filter the collection with.

        Returns:
            A new collection with the filtered simulations.
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
        return [sim.name for sim in self._orm.simulations]

    def sync_cache(self, *, force_all: bool = False) -> None:
        self._index.sync_collection(self.uid, self.path, force_all=force_all)

    def create_simulation(
        self,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        description: Optional[str] = None,
        files: Optional[Iterable[str]] = None,
        links: Optional[Dict[str, str]] = None,
        override: bool = False,
    ) -> SimulationWriter:
        """Get a writer object for a new simulation. This is written for paralell use
        as it is likely that this may be used in an executable, creating multiple runs
        for a parametric space, which may be run in paralell.

        Args:
            uid: The name/uid for the simulation. If not specified, a random id
                will be assigned.
            parameters: Parameter dictionary. If provided, the parameters will be
                checked against the existing sims for duplication. Otherwise, they may be
                specified later with `bamboost.core.simulation.Simulation.parameters`.

                Note:
                    The parameters are stored in the h5 file as attributes.
                    - If the value is a dict, it is flattened using
                      `bamboost.core.utilities.flatten_dict`.
                    - If the value is a list/array, it is stored as a dataset.

            skip_duplicate_check: if True, the duplicate check is skipped.
            prefix: Prefix for the uid. If not specified, no prefix is used.
            duplicate_action: how to deal with duplicates. Replace
                first duplicate ('r'), Create with altered uid (`c`), Create new
                with new id (`n`), Abort (`a`) default "prompt" for each
                duplicate on a case by case basis.
            note: Note for the simulation.
            files: List of files to copy to the simulation directory.
            links: Dictionary of links to other simulations.

        Note:
            The files and links are copied to the simulation directory. The files are
            copied with the same name as the original file. The links are copied with
            the given name.

        Examples:
            >>> db.create_simulation(parameters={"a": 1, "b": 2})

            >>> db.create_simulation(uid="my_sim", parameters={"a": 1, "b": 2}, prefix="test")

        Returns:
            A simulation writer object.
        """
        import shutil

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
            if exists and override:
                with RootProcessMeta.comm_self(self):
                    self._index._drop_simulation(self.uid, name)
                shutil.rmtree(directory)  # remove the old directory
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

    def _delete_simulation(self, name: str) -> None:
        """CAUTIOUS. Deletes a simulation.

        Args:
            name: Name of the simulation to delete.
        """
        dir_to_delete = self.path.joinpath(name)
        if dir_to_delete.parent != self.path:
            raise ValueError(f"Invalid name given ({name}). Cannot delete.")

        import shutil

        shutil.rmtree(dir_to_delete)

    def find(self, parameter_selection: dict[str, Any]) -> pd.DataFrame:
        """Find simulations with the given parameters.

        The dictionary can contain callables to filter inequalities or other
        filters.

        Examples:
            >>> db.find({"a": 1, "b": lambda x: x > 2})
            >>> db.find({"a": 1, "b": 2})

        Args:
            parameter_selection (dict): parameter selection dictionary
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
        self, parameters: dict, *, df: pd.DataFrame | None = None
    ) -> list[str]:
        """List ids of duplicates of the given parameters.

        Args:
            parameters: parameter dictionary
            df: dataframe to search in. If not provided, the
                dataframe from the sql database is used.
        """
        import pandas as pd

        if df is None:
            df = self._orm.to_pandas()
        params = flatten_dict(parameters)

        class ComparableIterable:
            def __init__(self, ori):
                self.ori = np.asarray(ori)

            def __eq__(self, other):
                other = np.asarray(other)
                if other.shape != self.ori.shape:
                    return False
                return (other == self.ori).all()

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

    def _check_duplicate(
        self, parameters: dict, uid: str, duplicate_action: str = "prompt"
    ) -> tuple:
        """Checking whether the parameters dictionary exists already.
        May need to be improved...

        Args:
            parameters: parameter dictionary to check for
            uid: uid

        Returns:
            Tuple(Bool, uid) wheter to continue and with what uid.
        """

        duplicates = self._list_duplicates(parameters)

        if not duplicates:
            return True, uid

        print(
            "The parameter space already exists. Here are the duplicates:",
            flush=True,
        )
        print(self.df[self.df["name"].isin([i for i in duplicates])], flush=True)

        if duplicate_action == "prompt":
            # What should be done?
            prompt = input(
                f"`r`: Replace first duplicate [{duplicates[0]}]\n"
                "`n`: Create new with new id\n"
                "`a`: Abort\n"
            )
        else:
            prompt = duplicate_action

        if prompt == "r":
            self._delete_simulation(duplicates[0])
            return True, uid
        if prompt == "a":
            return False, uid
        if prompt == "n":
            return True, uid

        raise ArgumentError("Answer not valid! Aborting")
