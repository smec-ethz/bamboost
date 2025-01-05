# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import inspect
import os
import subprocess
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property, wraps
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Optional,
    Tuple,
    Union,
)

import h5py
import numpy as np
from typing_extensions import Self, deprecated

import bamboost.core.simulation._repr as reprs
from bamboost import BAMBOOST_LOGGER, config
from bamboost._typing import (
    _MT,
    Immutable,
    Mutable,
    SimulationMetadataT,
    SimulationParameterT,
)
from bamboost.core import utilities
from bamboost.core.hdf5.dict import AttrsDict
from bamboost.core.hdf5.file import (
    FileMode,
    HDF5File,
    HDF5Path,
    with_file_open,
)
from bamboost.core.hdf5.ref import Group
from bamboost.core.simulation.dict import Links, Metadata, Parameters
from bamboost.core.simulation.groups import (
    PATH_DATA,
    PATH_FIELD_DATA,
    PATH_SCALAR_DATA,
    GroupData,
    GroupGit,
)
from bamboost.core.simulation.xdmf import XDMFWriter
from bamboost.index import (
    CollectionUID,
    Index,
)
from bamboost.mpi import MPI, MPI_ON
from bamboost.utilities import StrPath

if TYPE_CHECKING:
    import pandas as pd

    from bamboost.mpi import Comm


log = BAMBOOST_LOGGER.getChild("simulation")

UID_SEPARATOR = ":"
_HDFFileName = "data.h5"
_XDMFFileName = "data.xdmf"
_RUNFileName = "run.sh"


class SimulationName(str):
    """Name of a simulation."""

    def __new__(cls, name: Optional[str] = None, length: int = 10):
        name = name or cls.generate_name(length)
        return super().__new__(cls, name)

    @staticmethod
    def generate_name(length: int) -> str:
        return uuid.uuid4().hex[:length]


def _on_root(func):
    """Decorator to run a function only on the root process."""

    @wraps(func)
    def wrapper_with_bcast(self: Simulation, *args, **kwargs):
        res = None
        if self._comm.rank == 0:
            res = func(self, *args, **kwargs)
        return self._comm.bcast(res, root=0)

    @wraps(func)
    def wrapper(self: Simulation, *args, **kwargs):
        if self._comm.rank == 0:
            return func(self, *args, **kwargs)

    # check if return annotation is not None
    if inspect.signature(func).return_annotation is not None:
        return wrapper_with_bcast
    else:
        return wrapper


class _Simulation(ABC, Generic[_MT]):
    """Simulation accessor.

    Args:
        name: Name for the simulation.
        parent: Path to parent/collection directory.
        comm: MPI communicator. Defaults to MPI.COMM_WORLD.
        index: Index object. Defaults to the global index file.

    Raises:
        FileNotFoundError: If the simulation doesn't exist.
    """

    _mesh_location = "meshes"
    _default_mesh = "default"

    _repr_html_ = reprs.simulation_html_repr

    def __init__(
        self,
        name: str,
        parent: StrPath,
        comm: Optional[Comm] = None,
        index: Optional[Index] = None,
        **kwargs,
    ):
        self.name: str = name
        self.path: Path = Path(parent).joinpath(name).absolute()
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Simulation {self.name} does not exist in {self.path}."
            )
        self._index: Index = index or Index()

        # Shortcut to collection uid if available, otherwise resolve it
        self.collection_uid: CollectionUID = kwargs.pop(
            "collection_uid", None
        ) or self._index.resolve_uid(self.path.parent)

        self._data_file: Path = self.path.joinpath(_HDFFileName)
        self._xdmf_file: Path = self.path.joinpath(_XDMFFileName)
        self._bash_file: Path = self.path.joinpath(_RUNFileName)

        # MPI information
        self._comm: Comm = comm or MPI.COMM_WORLD
        self._psize: int = self._comm.size
        self._prank: int = self._comm.rank
        self._ranks = np.array([i for i in range(self._psize)])

        # Alias attributes
        self.root: Group[_MT] = self._file.root
        """Access to HDF5 file root group."""

        # Initialize groups to meshes, data and userdata. Create groups.
        # self.meshes: MeshGroup = MeshGroup(self._file)
        # self.data: DataGroup = DataGroup(self._file, self.meshes)
        # self.globals: GlobalGroup = GlobalGroup(self._file, "/globals")
        # self.userdata: hdf_pointer.MutableGroup = hdf_pointer.MutableGroup(
        #     self._file, "/userdata"
        # )

    @property
    @abstractmethod
    def _file(self) -> HDF5File[_MT]: ...

    @classmethod
    def from_uid(cls, uid: str, **kwargs) -> Self:
        """Return the `Simulation` with given UID.

        Args:
            uid: the full id (Collection uid : simulation name)
            **kwargs: additional arguments to pass to the constructor
        """
        collection_uid, name = uid.split(UID_SEPARATOR)
        index = kwargs.pop("index", None) or Index()
        collection_path = index.resolve_path(collection_uid)
        return cls(name, collection_path, index=index, **kwargs)

    @property
    def uid(self) -> str:
        """The full uid of the simulation (collection_uid:simulation_name)."""
        return f"{self.collection_uid}{UID_SEPARATOR}{self.name}"

    def edit(self) -> SimulationWriter:
        """Return an object with writing rights to edit the simulation."""
        return SimulationWriter(self.name, self.path.parent, self._comm, self._index)

    @_on_root
    def send_to_sql(
        self,
        *,
        metadata: Optional[SimulationMetadataT] = None,
        parameters: Optional[SimulationParameterT] = None,
    ) -> None:
        """Push update to sqlite database.

        Args:
            - update_dict (dict): key value pair to push
        """
        if not config.index.syncTables:
            return

        if metadata:
            self._index.update_simulation_metadata(
                self.collection_uid, self.name, metadata
            )
        if parameters:
            self._index.update_simulation_parameters(
                self.collection_uid, self.name, parameters
            )

    @cached_property
    def parameters(self) -> Parameters[_MT]:
        return Parameters(self)

    @cached_property
    def metadata(self) -> Metadata[_MT]:
        return Metadata(self)

    @cached_property
    def links(self) -> Links[_MT]:
        return Links(self)

    @cached_property
    def files(self):
        return utilities.FilePicker(self.path)

    @cached_property
    def git(self) -> GroupGit[_MT]:
        return GroupGit(self)

    def open_in_paraview(self) -> None:
        """Open the xdmf file in paraview."""
        subprocess.call(["paraview", self._xdmf_file])

    def create_xdmf_file(self, fields: list = None, nb_steps: int = None) -> None:
        """Create the xdmf file to read in paraview.

        Args:
            fields (list[str]): fields for which to write timeseries information,
                if not specified, all fields in data are written.
            nb_steps (int): number of steps the simulation has
        """
        # TODO: implement this method

        if self._prank == 0:
            with self._file("r"):
                f = self._file._h5py_file
                if "data" not in f.keys():
                    fields, nb_steps = [], 0
                if fields is None:
                    fields = list(f["data"].keys())

                if nb_steps is None:
                    grp_name = list(f["data"].keys())[0]
                    nb_steps = list(f[f"data/{grp_name}"].keys())
                    nb_steps = max(
                        [
                            int(step)
                            for step in nb_steps
                            if not (
                                step.startswith("__") or step.endswith("_intermediates")
                            )
                        ]
                    )

                # temporary fix to load coordinates/geometry
                coords_name = (
                    "geometry"
                    if "geometry"
                    in f[f"{self._mesh_location}/{self._default_mesh}"].keys()
                    else "coordinates"
                )

            with self._file("r"):
                xdmf_writer = XDMFWriter(self._xdmf_file, self._file)
                xdmf_writer.write_points_cells(
                    f"{self._mesh_location}/{self._default_mesh}/{coords_name}",
                    f"{self._mesh_location}/{self._default_mesh}/topology",
                )

                if fields:
                    xdmf_writer.add_timeseries(nb_steps + 1, fields)
                xdmf_writer.write_file()

        self._comm.barrier()

    def open(self, mode: FileMode | str = "r", driver=None) -> h5py.File:
        """Use this as a context manager in a `with` statement.
        Purpose: keeping the file open to directly access/edit something in the
        HDF5 file of this simulation.

        Args:
            mode: file mode (see h5py docs)
            driver: file driver (see h5py docs)
            comm: mpi communicator
        """
        mode = FileMode(mode)
        return self._file.open(mode, driver)

    @property
    def mesh(self) -> Mesh:
        """Return the default mesh.

        Returns:
            MeshGroup
        """
        return self.meshes[self._default_mesh]

    @with_file_open("r")
    def get_mesh(self, mesh_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return coordinates and connectivity. Currently returns numpy arrays.

        Args:
            mesh_name (`str`): optional, name of mesh to read (default = mesh)
        Returns:
            Tuple of np.arrays (coordinates, connectivity)
        """
        if mesh_name is None:
            mesh_name = self._default_mesh

        # Raise an error if the mesh is not found
        if (self._mesh_location.split("/")[0] not in self._file.keys()) or (
            mesh_name not in self._file[self._mesh_location].keys()
        ):
            raise KeyError(f"Mesh location {self._mesh_location} not found in file.")

        mesh = self.meshes[mesh_name]
        return mesh.coordinates, mesh.connectivity

    @property
    @deprecated("Use `data.info` instead")
    @with_file_open("r")
    def data_info(self) -> pd.DataFrame:
        """View the data stored.

        Returns:
            :class:`pd.DataFrame`
        """
        import pandas as pd

        tmp_dictionary = dict()
        for data in self.data:
            steps = len(data)
            shape = data.obj["0"].shape
            dtype = data.obj["0"].dtype
            tmp_dictionary[data._name] = {
                "dtype": dtype,
                "shape": shape,
                "steps": steps,
            }
        return pd.DataFrame.from_dict(tmp_dictionary)

    @with_file_open()
    def show_h5tree(self) -> None:
        """Print the tree inside the h5 file."""
        # print('\U00002B57 ' + os.path.basename(self.h5file))
        print("\U0001f43c " + os.path.basename(self._data_file))
        utilities.h5_tree(self._file.root._obj)

    @contextmanager
    def enter_path(self):
        """A context manager for changing the working directory to this simulations' path.

        >>> with sim.working_directory():
        >>>     ...
        """

        current_dir = os.getcwd()
        try:
            os.chdir(self.path)
            yield
        finally:
            os.chdir(current_dir)


class Simulation(_Simulation[Immutable]):
    @cached_property
    def _file(self) -> HDF5File[Immutable]:
        return HDF5File(self._data_file, comm=self._comm, mutable=False)


def dump_array_parallel(
    file: HDF5File[Mutable], path: HDF5Path, arr: np.ndarray
) -> None:
    dim = arr.shape[1:] if arr.ndim > 1 else ()
    len_local = arr.shape[0]
    len_others = np.array(file._comm.allgather(len_local))
    len_total = len_others.sum()

    # global indices
    idx_start = np.sum(len_others[: file._comm.rank])
    idx_end = idx_start + len_local

    with file.open(FileMode.APPEND, driver="mpio") as f:
        grp = f.require_group(path.parent)
        dataset = grp.require_dataset(
            path.basename, shape=(len_total, *dim), dtype=arr.dtype
        )
        dataset[idx_start:idx_end] = arr


class StepWriter:
    step: int

    def __init__(self, file: HDF5File[Mutable], step: int) -> None:
        self._file = file
        self._comm = file._comm
        self.step = step

    def dump_field_data(self, fieldname: str, data: np.ndarray) -> None:
        dump_array_parallel(
            self._file,
            HDF5Path(PATH_FIELD_DATA).joinpath(fieldname, str(self.step)),
            data,
        )

    @with_file_open(FileMode.APPEND, root_only=True)
    def dump_scalar_data(self, scalarname: str, data: Union[Number, Iterable]) -> None:
        data_arr = np.array(data)
        grp_scalars = self._file.require_group(PATH_SCALAR_DATA)

        # if the dataset does not exist, create it
        dataset = grp_scalars.require_dataset(
            scalarname,
            shape=(1, *data_arr.shape),
            dtype=data_arr.dtype,
            maxshape=(None, *data_arr.shape),
            chunks=True,
            fillvalue=np.full_like(data_arr, np.nan),
        )

        # append the data
        dataset.resize(self.step + 1, axis=0)
        dataset[self.step] = data_arr


class SimulationWriter(_Simulation[Mutable]):
    def __init__(
        self,
        name: str,
        parent: StrPath,
        comm: Optional[Comm] = None,
        index: Optional[Index] = None,
        **kwargs,
    ):
        super().__init__(name, parent, comm, index, **kwargs)

    def __enter__(self) -> SimulationWriter:
        self.metadata["status"] = "started"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.change_status(f"failed [{exc_type.__name__}]")
            log.error(
                f"Simulation failed with {exc_type.__name__}: {exc_val}\nTraceback: {exc_tb}"
            )
        self._comm.barrier()

    def require_step(self, step: Optional[int] = None) -> StepWriter:
        attrs_data = AttrsDict(self._file, PATH_DATA)
        step = max(attrs_data.get("last_step", -1) + 1, step or 0)
        print(f"step: {step}")
        attrs_data["last_step"] = step
        return StepWriter(self._file, step)

    @cached_property
    def _file(self) -> HDF5File[Mutable]:
        return HDF5File(self._data_file, comm=self._comm, mutable=True)

    def initialize(self) -> None:
        """Initialize the simulation."""
        # create the data file
        with self._file.open(FileMode.WRITE, root_only=True) as f:
            self.metadata.update(
                {
                    "status": "initialized",
                    "created_at": datetime.now(),
                }
            )
            # create groups
            f.create_group(".parameters")
            f.create_group(".links")
            f.create_group(".userdata")
            data_grp = f.create_group(".data")
            data_grp.create_group("field_data")
            data_grp.create_group("scalar_data")

        self._comm.barrier()

    def change_status(self, status: str) -> None:
        """Change status of simulation.

        Args:
            status (str): new status
        """
        self.metadata["status"] = status

    def copy_files(self, files: Iterable[StrPath]) -> None:
        """Copy files to the simulation folder.

        Args:
            files: list of files/directories to copy
        """
        import shutil

        for file in files:
            path = Path(file)
            if path.is_file():
                shutil.copy(path, self.path)
            elif path.is_dir():
                shutil.copytree(path, self.path)

    def create_run_script(
        self,
        commands: list[str],
        euler: bool = True,
        sbatch_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a batch job and put it into the folder.

        Args:
            commands: A list of strings being the user defined commands to run
            euler: If false, a local bash script will be written
            sbatch_kwargs: Additional sbatch arguments.
                This parameter allows you to provide additional arguments to the `sbatch` command
                when submitting jobs to a Slurm workload manager. The arguments should be provided
                in the format of a dict of sbatch option name and values.

                Use this parameter to specify various job submission options such as the number of
                tasks, CPU cores, memory requirements, email notifications, and other sbatch options
                that are not covered by default settings.
                By default, the following sbatch options are set:
                - `--output`: The output file is set to `<uid>.out`.
                - `--job-name`: The job name is set to `<full_uid>`.

                The following arguments should bring you far:
                - `--ntasks`: The number of tasks to run. This is the number of MPI processes to start.
                - `--mem-per-cpu`: The memory required per CPU core.
                - `--time`: The maximum time the job is allowed to run.
                - `--tmp`: Temporary scratch space to use for the job.
        """
        script = "#!/bin/bash\n\n"

        # Add sbatch options
        if euler:
            if sbatch_kwargs is None:
                sbatch_kwargs = {}

            sbatch_kwargs.setdefault(
                "--output", f"{self.path.joinpath(self.name + '.out')}"
            )
            sbatch_kwargs.setdefault("--job-name", self.uid)

            for key, value in sbatch_kwargs.items():
                script += f"#SBATCH {key}={value}\n"

        # Add environment variables
        script += "\n"
        script += (
            f"""COLLECTION_DIR=$(sqlite3 {config.index.databaseFile} "SELECT path FROM collections WHERE uid='{self.collection_uid}'")\n"""
            f"SIMULATION_DIR={self.path.as_posix()}\n"
            f"SIMULATION_ID={self.uid}\n\n"
        )
        script += "\n".join(commands)

        with self._bash_file.open("w") as file:
            file.write(script)

        self.metadata["submitted"] = False

    def run_simulation(self, executable: str = "bash") -> None:
        assert not MPI_ON, "This method is not available during MPI execution."

        if not self._bash_file.exists():
            raise FileNotFoundError(
                f"Run script {self._bash_file} does not exist. Create one with `create_run_script`."
            )

        env = os.environ.copy()
        _ = env.pop("BAMBOOST_MPI", None)  # remove bamboost MPI environment variable
        subprocess.run([executable, self._bash_file.as_posix()], env=env)
        log.info(f'Simulation "{self.name}" submitted.')
        self.metadata["submitted"] = True

    def submit_simulation(self) -> None:
        self.run_simulation(executable="sbatch")
