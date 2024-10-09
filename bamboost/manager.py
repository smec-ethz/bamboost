# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import numbers
import os
import pkgutil
import shutil
import uuid
from ctypes import ArgumentError
from typing import Any, Generator, Iterable, Union

import h5py
import numpy as np
import pandas as pd

from bamboost import BAMBOOST_LOGGER, index
from bamboost.common.file_handler import open_h5file
from bamboost.common.mpi import MPI
from bamboost.common.utilities import flatten_dict
from bamboost.index import DatabaseTable, IndexAPI, config
from bamboost.simulation import Simulation
from bamboost.simulation_writer import SimulationWriter

__all__ = [
    "Manager",
    "ManagerFromUID",
    "ManagerFromName",
]

log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


class ManagerFromUID(object):
    """Get a database by its UID. This is used for autocompletion in ipython."""

    def __init__(self) -> None:
        # or [] to circumvent Null type (MPI)
        ids = IndexAPI().fetch(f"SELECT id, path FROM dbindex") or []
        self.completion_keys = tuple(
            [f'{key} - {"..."+val[-25:] if len(val)>=25 else val}' for key, val in ids]
        )

    def _ipython_key_completions_(self):
        return self.completion_keys

    def __getitem__(self, key) -> Manager:
        key = key.split()[0]  # take only uid
        return Manager(uid=key, create_if_not_exist=False)


class ManagerFromName(object):
    """Get a database by its path/name. This is used for autocompletion in ipython."""

    def __init__(self) -> None:
        paths = IndexAPI().fetch("SELECT path FROM dbindex") or []
        self.completion_keys = tuple(paths)

    def _ipython_key_completions_(self):
        return self.completion_keys

    def __getitem__(self, key) -> Manager:
        return Manager(key, create_if_not_exist=False)


class Manager:
    """View of database.

    Args:
        path (`str`): path to the directory of the database. If doesn't exist,
            a new database will be created.
        comm (`MPI.Comm`): MPI communicator
        uid: UID of the database

    Attributes:
        FIX_DF: If False, the dataframe of the database is reconstructed every
            time it is accessed.
        fromUID: Access a database by its UID
        fromName: Access a database by its path/name

    Example:
        >>> db = Manager("path/to/db")
        >>> db.df # DataFrame of the database
    """

    FIX_DF = True
    fromUID: ManagerFromUID = ManagerFromUID()
    fromName: ManagerFromName = ManagerFromName()

    def __init__(
        self,
        path: str = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        uid: str = None,
        create_if_not_exist: bool = True,
    ):
        # provided uid has precedence
        if uid is not None:
            path = self._index.get_path(uid.upper())
            path = comm.bcast(path, root=0)
        self.path = path
        self.comm = comm

        # check if path exists
        if not os.path.isdir(path):
            if not create_if_not_exist:
                raise NotADirectoryError("Specified path is not a valid path.")
            log.info(f"Created new database ({path})")
            self._make_new(path)

        # retrieve the UID of the database from the id file
        # if not found, a new one is generated
        self.UID = uid or self._retrieve_uid()

        # Update the SQL table for the database
        try:
            with self._index.open():
                self._index.insert_path(self.UID, self.path)
                self._table.create_database_table()
                self._table.sync()
        except index.Error as e:
            log.warning(f"index error: {e}")

    def __getitem__(self, key: Union[str, int]) -> Simulation:
        """Returns the simulation in the specified row of the dataframe.

        Args:
            key: The simulation identifier (`str`) or the row index (`int`).
        Returns:
            The selected simulation object.
        """
        if isinstance(key, str):
            return self.sim(key)
        else:
            return self.sim(self.df.loc[key, "id"])

    def _repr_html_(self) -> str:
        """HTML repr for ipython/notebooks. Uses string replacement to fill the
        template code.
        """
        html_string = pkgutil.get_data(__name__, "html/manager.html").decode()
        icon = pkgutil.get_data(__name__, "html/icon.txt").decode()
        return (
            html_string.replace("$ICON", icon)
            .replace("$db_path", self.path)
            .replace("$db_uid", self.UID)
            .replace("$db_size", str(len(self)))
        )

    def __len__(self) -> int:
        return len(self.all_uids)

    def __iter__(self) -> Generator[Simulation, None, None]:
        for sim in self.sims():
            yield sim

    def _ipython_key_completions_(self):
        return self.all_uids

    @property
    def _index(self) -> IndexAPI:
        """The index which contains this database."""
        return IndexAPI()

    @property
    def _table(self) -> DatabaseTable:
        """The table in the sql database for this database."""
        return self._index.get_database_table(self.UID)

    def _retrieve_uid(self) -> str:
        """Get the UID of this database from the file tree."""
        try:
            return index.get_uid_from_path(self.path)
        except FileNotFoundError:
            pass

        log.warning("Database exists but no UID found. Generating new UID.")
        return self._make_new(self.path)

    def _make_new(self, path) -> str:
        """Initialize a new database."""
        from datetime import datetime

        # Create directory for database
        os.makedirs(path, exist_ok=True)

        # Assign a unique id to the database
        self.UID = f"{uuid.uuid4().hex[:10]}".upper()
        uid_file = os.path.join(path, f".BAMBOOST-{self.UID}")
        with open(uid_file, "a") as f:
            f.write(self.UID + "\n")
            f.write(f'Date of creation: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        os.chmod(uid_file, 0o444)  # read only for uid file

        log.info(f"Registered new database (uid = {self.UID})")
        self._index.insert_path(self.UID, path)
        return self.UID

    @property
    def all_uids(self) -> list:
        if not self.FIX_DF or not hasattr(self, "_all_uids"):
            self._all_uids = self._get_uids()
        return self._all_uids

    @all_uids.setter
    def all_uids(self, value: set | list):
        self._all_uids = value

    def _get_uids(self) -> list:
        """Get all simulation names in the database."""
        all_uids = list()
        for dir in os.listdir(self.path):
            if not os.path.isdir(os.path.join(self.path, dir)):
                continue
            if any(
                [i.endswith(".h5") for i in os.listdir(os.path.join(self.path, dir))]
            ):
                all_uids.append(dir)
        return all_uids

    @property
    def df(self) -> pd.DataFrame:
        """View of the database and its parametric space.

        Returns:
            :class:`pd.DataFrame`
        """
        if not hasattr(self, "_dataframe"):
            return self.get_view()
        if self.FIX_DF and self._dataframe is not None:
            return self._dataframe
        return self.get_view()

    def _get_parameters_for_uid(
        self, uid: str, include_linked_sims: bool = False
    ) -> dict:
        """Get the parameters for a given uid.

        Args:
            uid (`str`): uid of the simulation
            include_linked_sims (`bool`): if True, include the parameters of linked sims
        """
        h5file_for_uid = os.path.join(self.path, uid, f"{uid}.h5")
        tmp_dict = dict()

        with open_h5file(h5file_for_uid, "r") as f:
            if "parameters" in f.keys():
                tmp_dict.update(f["parameters"].attrs)
            if "additionals" in f.keys():
                tmp_dict.update({"additionals": dict(f["additionals"].attrs)})
            tmp_dict.update(f.attrs)

        if include_linked_sims:
            for linked, full_uid in self.sim(uid).links.attrs.items():
                sim = Simulation.fromUID(full_uid)
                tmp_dict.update(
                    {f"{linked}.{key}": val for key, val in sim.parameters.items()}
                )
        return tmp_dict

    def get_view(self, include_linked_sims: bool = False) -> pd.DataFrame:
        """View of the database and its parametric space. Read from the sql
        database. If `include_linked_sims` is True, the individual h5 files are
        scanned.

        Args:
            include_linked_sims: if True, include the parameters of linked sims

        Examples:
            >>> db.get_view()
            >>> db.get_view(include_linked_sims=True)
        """
        if include_linked_sims:
            return self.get_view_from_hdf_files(include_linked_sims=include_linked_sims)

        try:
            with self._table.open():
                self._table.sync()
                df = self._table.read_table()
        except index.Error as e:
            log.warning(f"index error: {e}")
            return self.get_view_from_hdf_files(include_linked_sims=include_linked_sims)

        if df.empty:
            return df
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])

        # Sort dataframe columns
        columns_start = ["id", "notes", "status", "time_stamp"]
        columns_start = [col for col in columns_start if col in df.columns]
        self._dataframe = df[[*columns_start, *df.columns.difference(columns_start)]]

        opts = config.get("options", {})
        if "sort_table_key" in opts:
            self._dataframe.sort_values(
                opts.get("sort_table_key", "id"),
                ascending=opts.get("sort_table_order", "asc") == "asc",
                inplace=True,
            )
        return self._dataframe

    def get_view_from_hdf_files(
        self, include_linked_sims: bool = False
    ) -> pd.DataFrame:
        """View of the database and its parametric space. Read from the h5
        files metadata.

        Args:
            include_linked_sims: if True, include the parameters of linked sims
        """
        all_uids = self._get_uids()
        data = list()

        for uid in all_uids:
            tmp_dict = self._get_parameters_for_uid(
                uid, include_linked_sims=include_linked_sims
            )
            data.append(tmp_dict)

        df = pd.DataFrame.from_records(data)
        if df.empty:
            return df
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])

        # Sort dataframe columns
        columns_start = ["id", "notes", "status", "time_stamp"]
        columns_start = [col for col in columns_start if col in df.columns]
        self._dataframe = df[[*columns_start, *df.columns.difference(columns_start)]]
        return self._dataframe

    @property
    def data_info(self) -> pd.DataFrame:
        """Return view of stored data for all simulations

        Returns:
            :class:`pd.DataFrame`
        """
        data = list()
        for uid in self.all_uids:
            h5file_for_uid = os.path.join(self.path, uid, f"{uid}.h5")
            with open_h5file(h5file_for_uid, "r") as file:
                try:
                    tmp_dict = dict()
                    tmp_dict = {
                        key: (
                            len(file[f"data/{key}"]),
                            file[f"data/{key}/0"].shape,
                            file[f"data/{key}/0"].dtype,
                        )
                        for key in file["data"].keys()
                    }
                except KeyError:
                    tmp_dict = dict()
                data.append(tmp_dict)
        return pd.DataFrame.from_records(data)

    def sim(
        self,
        uid: str,
        return_writer: bool = False,
        writer_type: SimulationWriter = SimulationWriter,
    ) -> Simulation:
        """Get an existing simulation with uid. Same as accessing with `db[uid]` directly.

        Args:
            uid (`str`): unique identifier
            return_writer: if true, return `SimulationWriter`, otherwise
                return `Simulation`
            writer_type: Optionally, you can specify a custom writer type to return.

        Returns:
            :class:`~bamboost.simulation.Simulation`
        """
        if return_writer:
            return writer_type(uid, self.path, self.comm)
        return Simulation(uid, self.path, self.comm, _db_id=self.UID)

    def sims(
        self,
        select: pd.Series | pd.DataFrame | dict = None,
        sort: str = None,
        reverse: bool = False,
        exclude: set = None,
        return_writer: bool = False,
    ) -> list[Simulation]:
        """Get all simulations in a list. Optionally, get all simulations matching the
        given selection using pandas.

        Args:
            select: Selection of simulations. Can be one of the following.
                - Pandas boolean series: A boolean series with the same length as the dataframe.
                - Pandas DataFrame: A subset of the full dataframe.
                - Dictionary: A dictionary with the parameters to select (see `find` for details).
            sort (`str`): Optionally sort the list with this keyword
            reverse (`bool`): swap sort direction
            exclude (`list[str]`): sims to exclude
            return_writer: if true, return `SimulationWriter`, otherwise
                return `Simulation`

        Returns:
            A list of `:class:~bamboost.simulation.Simulation` objects

        Examples:
            >>> db.sims(select=db.df["status"] == "finished", sort="time_stamp")
        """
        if select is None:
            id_list = self.all_uids
        elif isinstance(select, pd.DataFrame):
            id_list = select["id"].values
        elif isinstance(select, pd.Series):
            id_list = self.df[select]["id"].values
        elif isinstance(select, dict):
            id_list = self.find(select)["id"].values
        else:
            raise ArgumentError('Invalid argument for argument "select"')

        if exclude is not None:
            exclude = list([exclude]) if isinstance(exclude, str) else exclude
            id_list = [id for id in id_list if id not in exclude]

        existing_sims = [self.sim(uid, return_writer) for uid in id_list]

        if sort is None:
            return existing_sims
        else:
            return sorted(
                existing_sims, key=lambda s: s.parameters[sort], reverse=reverse
            )

    def create_simulation(
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
        """
        if parameters and not skip_duplicate_check:
            go_on = True
            if self.comm.rank == 0:
                go_on, uid = self._check_duplicate(
                    parameters, uid, duplicate_action=duplicate_action
                )
            self.comm.bcast((go_on, uid), root=0)
            if not go_on:
                print("Aborting by user desire...")
                return None

        if self.comm.rank == 0:
            if not uid:
                uid = uuid.uuid4().hex[:8]  # Assign random unique identifier
            if isinstance(prefix, str) and prefix != "":
                uid = "_".join([prefix, uid])
        uid = self.comm.bcast(uid, root=0)

        try:
            # Create directory and h5 file
            if self.comm.rank == 0:
                os.makedirs(os.path.join(self.path, uid), exist_ok=True)
                path_to_h5_file = os.path.join(self.path, uid, f"{uid}.h5")
                if os.path.exists(path_to_h5_file):
                    os.remove(path_to_h5_file)
                h5py.File(path_to_h5_file, "a").close()  # create file

            new_sim = SimulationWriter(uid, self.path, self.comm)
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
            if self.comm.rank == 0:
                self.remove(uid)
            raise e  # Re-raise the exception after cleanup

    def remove(self, uid: str) -> None:
        """CAUTION, DELETING DATA. Remove the data of a simulation.

        Args:
            uid (`str`): uid
        """
        shutil.rmtree(os.path.join(self.path, uid))
        self._table.sync()

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
                "Replace first duplicate ('r'), Create with altered uid (`c`), "
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

    def _generate_subuid(self, uid_base: str) -> str:
        """Return a new sub uid for the base uid.
        Following the following format: `base_uid.1`

        Args:
            uid_base (`str`): base uid for which to find the next subid.
        Returns:
            New uid string
        """
        uid_list = [uid for uid in self.all_uids if uid.startswith(uid_base)]
        subiterator = max(
            [int(id.split(".")[1]) for id in uid_list if len(id.split(".")) > 1] + [0]
        )
        return f"{uid_base}.{subiterator+1}"

    def global_fields_in_all(self) -> list:
        """Get a list of all global fields in all simulations.

        Returns:
            List of global fields
        """
        fields = set()
        for sim in self:
            try:
                fields.update(sim.globals.columns)
            except KeyError:
                continue

        return fields

    def get_parameters(self) -> dict:
        """Get the parameters used in this database.

        Returns:
            Dictionary of parameters with it's count, range, and type. Sorted by count.
        """
        parameters = dict()
        for sim in self:
            for key, val in sim.parameters.items():
                if key not in parameters:
                    range = (val, val) if isinstance(val, numbers.Number) else None
                    parameters[key] = {"range": range, "count": 1, "type": type(val)}
                else:
                    if isinstance(val, numbers.Number):
                        parameters[key]["range"] = (
                            min(parameters[key]["range"][0], val),
                            max(parameters[key]["range"][1], val),
                        )
                    parameters[key]["count"] += 1
                    parameters[key]["type"] = type(val)
        return dict(
            sorted(parameters.items(), key=lambda x: x[1]["count"], reverse=True)
        )
