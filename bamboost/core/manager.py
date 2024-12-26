# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import os
import pkgutil
import uuid
from ctypes import ArgumentError
from functools import cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Optional,
)

import h5py
import numpy as np

from bamboost import BAMBOOST_LOGGER, config

# from bamboost.core.simulation.base import Simulation
# from bamboost.core.simulation.writer import SimulationWriter
from bamboost.core.simulation.base import Simulation
from bamboost.core.utilities import flatten_dict
from bamboost.index import CollectionUID, Index, StrPath
from bamboost.mpi import MPI

if TYPE_CHECKING:
    import pandas as pd

    from bamboost.mpi import Comm


log = BAMBOOST_LOGGER.getChild("Collection")


class NotACollectionError(NotADirectoryError):
    """Raised when a path is not a valid collection."""

    def __init__(self, path: Path):
        super().__init__(f"{path} is not a valid collection.")


class Collection:
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

    def __init__(
        self,
        path: Optional[StrPath] = None,
        *,
        uid: Optional[str] = None,
        create_if_not_exist: bool = True,
        comm: Optional[Comm] = None,
        index_instance: Optional[Index] = None,
    ):
        assert path or uid, "Either path or uid must be provided."
        assert not (path and uid), "Only one of path or uid must be provided."

        self._comm = comm or MPI.COMM_WORLD
        self._index = index_instance or Index()

        # Resolve the path
        self.path = Path(path or self._index.resolve_path(uid.upper()))
        if not self.path.is_dir():
            if not create_if_not_exist:
                raise NotADirectoryError("Specified path does not exist.")

            self.path.mkdir(parents=True, exist_ok=False)  # create the directory
            log.info(f"Initialized directory for collection at {path}")

        # Resolve or get an UID for the collection
        self.uid = CollectionUID(uid or self._index.resolve_uid(self.path))

        # Sync the SQL table with the filesystem
        # Making sure the collection is up to date in the index
        self._index.sync_collection(self.uid, self.path)

    @property
    def _orm(self):
        return self._index.collection(self.uid)

    def __len__(self) -> int:
        return len(self._orm.simulations)

    def __getitem__(self, name: str) -> Simulation:
        return Simulation(name, self.path, self._comm)

    @cache
    def _ipython_key_completions_(self):
        return tuple(s.name for s in self._orm.simulations)

    def _repr_html_(self) -> str:
        """HTML repr for ipython/notebooks. Uses string replacement to fill the
        template code.
        """
        html_string = pkgutil.get_data("bamboost", "_repr/manager.html").decode()
        icon = pkgutil.get_data("bamboost", "_repr/icon.txt").decode()
        return (
            html_string.replace("$ICON", icon)
            .replace("$db_path", f"<a href={self.path.as_posix()}>{self.path}</a>")
            .replace("$db_uid", self.uid)
            .replace("$db_size", str(len(self)))
        )

    @property
    def df(self) -> pd.DataFrame:
        """View of the collection and its parametric space.

        Returns:
            A dataframe of the collection
        """
        import pandas as pd

        df = pd.DataFrame.from_records(
            [sim.as_dict(standalone=False) for sim in self._orm.simulations]
        )
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

    def sync_cache(self):
        self._index.sync_collection(self.uid, self.path)

    def create_simulation(
        self,
    ) -> Simulation:
        pass

    def create_simulation_old(
        self,
        uid: str = None,
        parameters: dict = None,
        skip_duplicate_check: bool = False,
        *,
        prefix: str = None,
        duplicate_action: str = "prompt",
        note: str = None,
        files: list[str] = None,
        links: dict[str, str] = None,
    ) -> SimulationWriter:
        """Get a writer object for a new simulation. This is written for paralell use
        as it is likely that this may be used in an executable, creating multiple runs
        for a parametric space, which may be run in paralell.

        Args:
            uid (`str`): The name/uid for the simulation. If not specified, a random id
                will be assigned.
            parameters (`dict`): Parameter dictionary. If provided, the parameters will be
                checked against the existing sims for duplication. Otherwise, they may be
                specified later with `bamboost.simulation_writer.SimulationWriter.add_parameters`.
                Note:
                    The parameters are stored in the h5 file as attributes.
                    - If the value is a dict, it is flattened using
                      `bamboost.common.utilities.flatten_dict`.
                    - If the value is a list/array, it is stored as a dataset.
            skip_duplicate_check (`bool`): if True, the duplicate check is skipped.
            prefix (`str`): Prefix for the uid. If not specified, no prefix is used.
            duplicate_action (`str`): how to deal with duplicates. Replace
                first duplicate ('r'), Create with altered uid (`c`), Create new
                with new id (`n`), Abort (`a`) default "prompt" for each
                duplicate on a case by case basis.
            note (`str`): Note for the simulation.
            files (`list`): List of files to copy to the simulation directory.
            links (`dict`): Dictionary of links to other simulations.

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
        if parameters and not skip_duplicate_check:
            go_on = True
            if self._comm.rank == 0:
                go_on, uid = self._check_duplicate(
                    parameters, uid, duplicate_action=duplicate_action
                )
            self._comm.bcast((go_on, uid), root=0)
            if not go_on:
                print("Aborting by user desire...")
                return None

        if self._comm.rank == 0:
            if not uid:
                uid = uuid.uuid4().hex[:8]  # Assign random unique identifier
            if isinstance(prefix, str) and prefix != "":
                uid = "_".join([prefix, uid])
        uid = self._comm.bcast(uid, root=0)

        try:
            # Create directory and h5 file
            if self._comm.rank == 0:
                os.makedirs(os.path.join(self.path, uid), exist_ok=True)
                path_to_h5_file = os.path.join(self.path, uid, f"{uid}.h5")
                if os.path.exists(path_to_h5_file):
                    os.remove(path_to_h5_file)
                h5py.File(path_to_h5_file, "a").close()  # create file

            new_sim = SimulationWriter(uid, self.path, self._comm)
            new_sim.initialize()  # sets metadata and status
            # add the id to the (fixed) _all_uids list
            if hasattr(self, "_all_uids"):
                self._all_uids.append(new_sim.uid)

            # Add parameters, note, files, and links
            if not any([parameters, note, files, links]):
                return new_sim

            with new_sim._file("r+"):
                if parameters:
                    new_sim.add_parameters(parameters)
                if note:
                    new_sim.change_note(note)
                if files:
                    new_sim.copy_file(files)
                if links:
                    [
                        new_sim.links.__setitem__(name, uid)
                        for name, uid in links.items()
                    ]

            return new_sim

        except Exception as e:
            # If any error occurs, remove the partially created simulation
            if self._comm.rank == 0:
                self.remove(uid)
            raise e  # Re-raise the exception after cleanup

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
        matches = df[df.id.isin(matches)]
        if len(matches) == 0:
            return matches

        for key, func in filters.items():
            matches = matches[matches[key].apply(func)]

        return matches

    def _list_duplicates(
        self, parameters: dict, *, df: pd.DataFrame = None
    ) -> list[str]:
        """List ids of duplicates of the given parameters.

        Args:
            parameters (dict): parameter dictionary
            df (pd.DataFrame): dataframe to search in. If not provided, the
                dataframe from the sql database is used.
        """
        if df is None:
            df: pd.DataFrame = self._table.read_table()
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
        return match.id.tolist()

    def _check_duplicate(
        self, parameters: dict, uid: str, duplicate_action: str = "prompt"
    ) -> tuple:
        """Checking whether the parameters dictionary exists already.
        May need to be improved...

        Args:
            parameters (`dict`): parameter dictionary to check for
            uid (`str`): uid
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
        print(self.df[self.df["id"].isin([i for i in duplicates])], flush=True)

        if duplicate_action == "prompt":
            # What should be done?
            prompt = input(
                f"Replace first duplicate [{duplicates[0]}] ('r'), Create with altered uid (`c`), "
                + "Create new with new id (`n`), Abort (`a`): "
            )
        else:
            prompt = duplicate_action

        if prompt == "r":
            self.remove(duplicates[0])
            return True, uid
        if prompt == "a":
            return False, uid
        if prompt == "n":
            return True, uid
        if prompt == "c":
            return True, self._generate_subuid(duplicates[0].split(".")[0])

        raise ArgumentError("Answer not valid! Aborting")
