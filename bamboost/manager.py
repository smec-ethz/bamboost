# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
from __future__ import annotations

import logging
import numbers
import os
import pkgutil
import shutil
import uuid
from ctypes import ArgumentError
from typing import Union

import h5py
import pandas as pd
from mpi4py import MPI

from . import index
from .common.file_handler import open_h5file
from .simulation import Simulation
from .simulation_writer import SimulationWriter

__all__ = ["Manager", "ManagerFromUID", "ManagerFromName"]

log = logging.getLogger(__name__)


META_INFO = """
This database has been created using `bamboost`, a python package developed at
the CMBM group of ETH zurich. It has been built for data management using the
HDF5 file format.

https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
"""

# Setup Manager getters
# ---------------------


class ManagerFromUID(object):
    """Get a database by its UID. This is used for autocompletion in ipython."""

    def __init__(self) -> None:
        ids = index.get_index_dict()
        self.completion_keys = tuple(
            [
                f'{key} - {"..."+val[-25:] if len(val)>=25 else val}'
                for key, val in ids.items()
            ]
        )

    def _ipython_key_completions_(self):
        return self.completion_keys

    def __getitem__(self, key) -> Manager:
        key = key.split()[0]  # take only uid
        return Manager(uid=key, create_if_not_exist=False)


class ManagerFromName(object):
    """Get a database by its path/name. This is used for autocompletion in ipython."""

    def __init__(self) -> None:
        self.completion_keys = tuple(index.get_index_dict().values())

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
        if uid is not None:
            path = index.get_path(uid.upper())
        self.path = path
        self.comm = comm

        # check if path exists
        if not os.path.isdir(path):
            if not create_if_not_exist:
                raise NotADirectoryError("Specified path is not a valid path.")
            log.info(f"Created new database ({path})")
            self._make_new(path)
        self.UID = self._retrieve_uid()
        # self._store_uid_in_index()
        self._all_uids = self._get_uids()
        self._dataframe: pd.DataFrame = None
        self._meta_folder = os.path.join(path, ".database")

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

    def __iter__(self) -> Simulation:
        for sim in self.sims():
            yield sim

    def _ipython_key_completions_(self):
        return self.all_uids

    def _retrieve_uid(self) -> str:
        """Get the UID of this database from the file tree."""
        for file in os.listdir(self.path):
            if file.startswith(".BAMBOOST"):
                return file.split("-")[1]
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
        self._store_uid_in_index()
        return self.UID

    def _store_uid_in_index(self) -> None:
        """Stores the UID of this database with the current path."""
        index.record_database(self.UID, os.path.abspath(self.path))

    def _init_meta_folder(self) -> None:
        os.makedirs(self._meta_folder, exist_ok=True)
        with open(os.path.join(self._meta_folder, "README.txt"), "w") as f:
            f.write(META_INFO)

    @property
    def all_uids(self) -> set:
        if self.FIX_DF:
            return self._all_uids
        return self._get_uids()

    @property
    def df(self) -> pd.DataFrame:
        """View of the database and its parametric space.

        Returns:
            :class:`pd.DataFrame`
        """
        if self.FIX_DF and self._dataframe is not None:
            return self._dataframe
        return self.get_view()

    def get_view(self, include_linked_sims: bool = False) -> pd.DataFrame:
        """View of the database and its parametric space.

        Args:
            include_linked_sims: if True, include the parameters of linked sims

        Returns:
            :class:`pd.DataFrame`
        """
        all_uids = self.all_uids
        data = list()

        for uid in all_uids:
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
            data.append(tmp_dict)

        df = pd.DataFrame.from_records(data)
        if df.empty:
            return df
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])

        # Sort dataframe columns
        columns_start = ["id", "notes", "status", "time_stamp"]
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
                tmp_dict = dict()
                tmp_dict = {
                    key: (
                        len(file[f"data/{key}"]),
                        file[f"data/{key}/0"].shape,
                        file[f"data/{key}/0"].dtype,
                    )
                    for key in file["data"].keys()
                }
                data.append(tmp_dict)
        return pd.DataFrame.from_records(data)

    def sim(self, uid, return_writer: bool = False) -> Simulation:
        """Get an existing simulation with uid. Same as accessing with `db[uid]` directly.

        Args:
            uid (`str`): unique identifier
            return_writer: if true, return `SimulationWriter`, otherwise
                return `Simulation`
        Returns:
            :class:`~bamboost.simulation.Simulation`
        """
        if uid not in self.all_uids:
            raise KeyError("The simulation id is not valid.")
        if return_writer:
            return SimulationWriter(uid, self.path, self.comm)
        return Simulation(uid, self.path, self.comm)

    def sims(
        self,
        select: pd.Series = None,
        sort: str = None,
        reverse: bool = False,
        exclude: set = None,
        return_writer: bool = False,
    ) -> list:
        """Get all simulations in a list. Optionally, get all simulations matching the
        given selection using pandas.

        Args:
            select (`pd.Series`): pandas boolean series
            sort (`str`): Optionally sort the list with this keyword
            reverse (`bool`): swap sort direction
            exclude (`list[str]`): sims to exclude
            return_writer: if true, return `SimulationWriter`, otherwise
                return `Simulation`
        Returns:
            A list of `:class:~bamboost.simulation.Simulation` objects
        """
        if select is not None:
            id_list = self.df[select]["id"].values
        else:
            id_list = self.all_uids
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
        prefix: str = None,
    ) -> SimulationWriter:
        """Get a writer object for a new simulation. This is written for paralell use
        as it is likely that this may be used in an executable, creating multiple runs
        for a parametric space, which may be run in paralell.

        Args:
            uid (`str`): The name/uid for the simulation. If not specified, a random id
                will be assigned.
            parameters (`dict`): Parameter dictionary. If provided, the parameters will be
                checked against the existing sims for duplication. Otherwise, they may be
                specified later with :func:`~bamboost.simulation.SimulationWriter.add_parameters`.
            skip_duplicate_check (`bool`): if True, the duplicate check is skipped.
            prefix (`str`): Prefix for the uid. If not specified, no prefix is used.
        Returns:
            sim (:class:`~bamboost.simulation.SimulationWriter`)
        """
        if parameters and not skip_duplicate_check:
            go_on, uid = self._check_duplicate(parameters, uid)
            if not go_on:
                print("Aborting by user desire...")
                return None

        if self.comm.rank == 0:
            if not uid:
                uid = uuid.uuid4().hex[:8]  # Assign random unique identifier
            if isinstance(prefix, str) and prefix != "":
                uid = "_".join([prefix, uid])
        uid = self.comm.bcast(uid, root=0)

        # Create directory and h5 file
        if self.comm.rank == 0:
            os.makedirs(os.path.join(self.path, uid), exist_ok=True)
            path_to_h5_file = os.path.join(self.path, uid, f"{uid}.h5")
            if os.path.exists(path_to_h5_file):
                os.remove(path_to_h5_file)
            h5py.File(path_to_h5_file, "a").close()  # create file

        new_sim = SimulationWriter(uid, self.path, self.comm)
        new_sim.initialize()  # sets metadata and status
        self.all_uids.append(new_sim.uid)
        if parameters is None:
            parameters = dict()
        new_sim.add_parameters(parameters)
        return new_sim

    def remove(self, uid: str) -> None:
        """CAUTION, DELETING DATA. Remove the data of a simulation.

        Args:
            uid (`str`): uid
        """
        shutil.rmtree(os.path.join(self.path, uid))

    def _get_uids(self) -> list:
        """Get all simulation names in the database."""
        all_uids = list()
        for dir in [i for i in os.listdir(self.path) if not i.startswith(".")]:
            if any(
                [i.endswith(".h5") for i in os.listdir(os.path.join(self.path, dir))]
            ):
                all_uids.append(dir)
        return all_uids

    def _check_duplicate(self, parameters: dict, uid: str) -> tuple:
        """Checking whether the parameters dictionary exists already.
        May need to be improved...

        Args:
            parameters (`dict`): parameter dictionary to check for
            uid (`str`): uid
        Returns:
            Tuple(Bool, uid) wheter to continue and with what uid.
        """
        duplicates = list()

        for _uid in self.all_uids:
            with open_h5file(
                os.path.join(os.path.join(self.path, _uid), f"{_uid}.h5"), "r"
            ) as f:
                if "parameters" not in f.keys():
                    continue

                tmp_dict = dict()
                tmp_dict.update(f["parameters"].attrs)

                if tmp_dict == parameters:
                    duplicates.append((_uid, "equal"))
                    continue

                shared_keys = tmp_dict.keys() & parameters.keys()
                if {key: tmp_dict[key] for key in shared_keys} == {
                    key: parameters[key] for key in shared_keys
                }:
                    duplicates.append((_uid, "shared_equal"))
                    continue

        if not duplicates:
            return True, uid

        # What should be done?
        print(
            f"The parameter space may already exist. Here are the duplicates:",
            flush=True,
        )
        print(self.df[self.df["id"].isin([i[0] for i in duplicates])], flush=True)

        prompt = input(
            "Replace first duplicate ('r'), Create with altered uid (`c`), "
            + "Create new with new id (`n`), Abort (`a`): "
        )
        if prompt == "r":
            self.remove(duplicates[0][0])
            return True, uid
        if prompt == "a":
            return False, uid
        if prompt == "n":
            return True, uid
        if prompt == "c":
            return True, self._generate_subuid(duplicates[0][0].split(".")[0])

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
