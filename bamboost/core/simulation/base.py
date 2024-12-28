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
from collections.abc import Mapping
from contextlib import contextmanager
from functools import cached_property, wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import h5py
import numpy as np
import pandas as pd
from typing_extensions import deprecated

import bamboost.core.simulation._repr as reprs
from bamboost import BAMBOOST_LOGGER, config
from bamboost.core import utilities
from bamboost.core.hdf5.file import (
    FileMode,
    Group,
    H5Reference,
    HDF5File,
    MutableGroup,
    with_file_open,
)
from bamboost.core.simulation.xdmf import XDMFWriter
from bamboost.index import (
    CollectionUID,
    Index,
    _SimulationMetadataT,
    _SimulationParameterT,
)
from bamboost.mpi import MPI, MPI_ON
from bamboost.utilities import StrPath

if TYPE_CHECKING:
    from bamboost.mpi import Comm


log = BAMBOOST_LOGGER.getChild("simulation")

UID_SEPARATOR = ":"

_GT = TypeVar("_GT", bound=H5Reference)


def _mutable_only(func):
    """Decorator to raise an error if the object is not mutable."""

    @wraps(func)
    def wrapper(self: _GroupDict, *args, **kwargs):
        if self._file._readonly:
            raise PermissionError("Simulation is read-only.")
        return func(self, *args, **kwargs)

    return wrapper


class _GroupDict(Mapping):
    """A dictionary-like object for the attributes of a group in the HDF5
    file.

    This object is tied to a simulation. If the simulation is read-only, the
    object is immutable. If mutable, changes are pushed to the HDF5 file
    immediately.

    Args:
        simulation: the simulation object
        path: path to the group in the HDF5 file
    """

    mutable: bool = False

    def __init__(self, simulation: Simulation, path: str) -> None:
        self._simulation = simulation
        self._path = path
        self._file = simulation._file
        self._dict = self.read()
        self.mutable = not self._file._readonly

    @with_file_open(FileMode.READ)
    def read(self) -> dict:
        tmp_dict = dict()

        try:
            grp = cast(h5py.Group, self._file[self._path])
        except KeyError:
            raise KeyError(
                f"Group {self._path} not found in file {self._simulation._file._filename}."
            )

        tmp_dict.update(grp.attrs)
        for key, value in grp.items():
            if not isinstance(value, h5py.Dataset):
                continue
            tmp_dict.update({key: value[()]})

        return utilities.unflatten_dict(tmp_dict)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self) -> str:
        return self._dict.__repr__()

    def _repr_pretty_(self, p, cycle):
        cls_name = type(self).__name__
        if cycle:
            p.text(f"{cls_name}(...)")
        else:
            with p.group(8, f"{cls_name}(", ")"):
                p.pretty(self._dict)

    def _ipython_key_completions_(self):
        return tuple(self._dict.keys())

    @property
    def _obj(self) -> h5py.Group:
        obj = self._file[self._path]
        assert isinstance(obj, h5py.Group), f"Object at {self._path} is not a group."
        return obj

    # Methods for mutable objects only
    @_mutable_only
    def __setitem__(self, key: str, value: Any) -> None:
        self._dict[key] = value
        with self._file.open(FileMode.APPEND):
            self._obj.attrs[key] = value

    @_mutable_only
    def __delitem__(self, key: str) -> None:
        with self._file.open(FileMode.APPEND):
            del self._obj.attrs[key]


class _Parameters(_GroupDict):
    def __init__(self, simulation: Simulation) -> None:
        super().__init__(simulation, "/parameters")


class _MutableParameters(_Parameters):
    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        # also send the updated parameter to the SQL database
        self._simulation._send_to_sql(parameters={key: value})

    def update(self, update_dict: dict) -> None:
        """Update the parameters dictionary. This method pushes the update to
        the HDF5 file, and the SQL database.

        Args:
            update_dict: new parameters
        """
        # update dictionary in memory
        self._dict.update(update_dict)

        # try update the sql database
        self._simulation._send_to_sql(parameters=update_dict)

        # Filter out numpy arrays
        arrays = {}
        for k, v in update_dict.items():
            if isinstance(v, np.ndarray):
                arrays[k] = update_dict.pop(k)

        with self._file.open(FileMode.APPEND):
            # write arrays as datasets
            for k, v in arrays.items():
                if k in self._obj:
                    del self._obj[k]
                self._obj.create_dataset(k, data=v)

            # write the rest
            self._obj.attrs.update(update_dict)


class _Links(_GroupDict):
    def __init__(self, simulation: Simulation) -> None:
        super().__init__(simulation, "/links")


class _Metadata(_GroupDict):
    def __init__(self, simulation: Simulation) -> None:
        super().__init__(simulation, "/")

    @_mutable_only
    def __setitem__(self, key: str, value: Any) -> None:
        assert (
            key in self._dict
        ), f'Can only set existing metadata keys. "{key}" not found.'
        super().__setitem__(key, value)
        # also send the updated parameter to the SQL database
        self._simulation._send_to_sql(metadata={key: value})  # type: ignore


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


class Simulation:
    """Simulation accessor.

    Args:
        name: Unique identifier for the simulation.
        path: Path to parent/database folder.
        comm: MPI communicator. Defaults to MPI.COMM_WORLD.
        create_if_not_exists: Create the simulation if it doesn't exist. Defaults to False.

    Raises:
        FileNotFoundError: If the simulation doesn't exist and create_if_not_exists is False.
    """

    _mesh_location = "Mesh/0"
    _default_mesh = "mesh"

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
        self._index: Index = index or Index()
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Simulation {self.name} does not exist in {self.path}."
            )

        # MPI information
        self._comm: Comm = comm or MPI.COMM_WORLD
        self._psize: int = self._comm.size
        self._prank: int = self._comm.rank
        self._ranks = np.array([i for i in range(self._psize)])

        self.collection_uid: CollectionUID = kwargs.pop(
            "collection_uid", None
        ) or self._index.resolve_uid(self.path.parent)

        self._data_file: Path = self.path.joinpath(f"{self.name}.h5")
        self._xdmf_file: Path = self.path.joinpath(f"{self.name}.xdmf")

        self._file = HDF5File(self._data_file, comm=self._comm, readonly=True)

        # Initialize groups to meshes, data and userdata. Create groups.
        # self.meshes: MeshGroup = MeshGroup(self._file)
        # self.data: DataGroup = DataGroup(self._file, self.meshes)
        # self.globals: GlobalGroup = GlobalGroup(self._file, "/globals")
        # self.userdata: hdf_pointer.MutableGroup = hdf_pointer.MutableGroup(
        #     self._file, "/userdata"
        # )

    @classmethod
    def from_uid(cls, uid: str, **kwargs) -> Simulation:
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

    @property
    def root(self) -> H5Reference:
        """Access to HDF5 file root group.

        Returns:
            A reference to the root HDF5 object.
        """
        return self._file.root

    @_on_root
    def _send_to_sql(
        self,
        *,
        metadata: Optional[_SimulationMetadataT] = None,
        parameters: Optional[_SimulationParameterT] = None,
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
    def parameters(self) -> _Parameters:
        return _Parameters(self)

    @cached_property
    def metadata(self) -> _Metadata:
        return _Metadata(self)

    @cached_property
    def links(self) -> _Links:
        return _Links(self)

    @cached_property
    def files(self):
        class FilePicker:
            def __init__(self, path: Path):
                self.path = path
                self._dict = self._build_file_dict(path)

            def _build_file_dict(self, path: Path) -> dict:
                file_dict = {}
                for f in path.iterdir():
                    if f.is_dir():
                        subdir_dict = self._build_file_dict(f)
                        file_dict.update(
                            {f"{f.name}/{k}": v for k, v in subdir_dict.items()}
                        )
                    else:
                        file_dict[f.name] = f.absolute()
                return file_dict

            def __getitem__(self, key):
                return self._dict[key]

            def _ipython_key_completions_(self):
                return tuple(self._dict.keys())

            def __repr__(self):
                return utilities.tree(self.path)

        return FilePicker(self.path)

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

    @property
    @with_file_open("r")
    def git(self) -> dict:
        """Get Git information.

        Returns:
            :class:`dict` with different repositories.
        """
        if "git" not in self._file.keys():
            return "Sorrrry, no git information stored :()"
        grp = self._file["git"]
        tmp_dict = {}
        for repo in grp.keys():
            tmp_dict[repo] = grp[repo][()].decode("utf8")
        return tmp_dict

    def get_data_interpolator(self, field: str, step: int):
        """Get Linear interpolator for data field at step. Uses the linked mesh.

        Args:
            name (`str`): name of the data field
            step (`int`): step
        Returns:
            :class:`scipy.interpolate.LinearNDInterpolator`
        """
        from scipy.interpolate import LinearNDInterpolator

        return LinearNDInterpolator(
            self.data[field].mesh.coordinates, self.data[field].at_step(step)
        )

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


class SimulationWriter(Simulation):
    def __init__(
        self,
        name: str,
        parent: StrPath,
        comm: Optional[Comm] = None,
        index: Optional[Index] = None,
        **kwargs,
    ):
        super().__init__(name, parent, comm, index, **kwargs)

        # Set the file to editable
        self._file._readonly = False

    @cached_property
    def parameters(self) -> _MutableParameters:
        return _MutableParameters(self)

    @_on_root
    @with_file_open(FileMode.APPEND)
    def change_status(self, status: str) -> None:
        """Change status of simulation.

        Args:
            status (str): new status
        """
        self._file.attrs["status"] = status
        self._send_to_sql(metadata={"status": status})

    @_on_root
    @with_file_open(FileMode.APPEND)
    def update_metadata(self, update_dict: _SimulationMetadataT) -> None:
        """Update the metadata attributes.

        Args:
            update_dict: dictionary to push
        """
        self._file.attrs.update(update_dict)
        self._send_to_sql(metadata=update_dict)

    @_on_root
    @with_file_open(FileMode.APPEND)
    def update_parameters(self, update_dict: _SimulationParameterT) -> None:
        """Update the parameters dictionary.

        Args:
            update_dict: dictionary to push
        """
        update_dict = utilities.flatten_dict(update_dict)
        self._file["parameters"].attrs.update(update_dict)
        self._send_to_sql(parameters=update_dict)

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

        with self.path.joinpath(f"{self.name}.sh").open("w") as file:
            file.write(script)

        self.metadata["submitted"] = False

    def sbatch_submit(self) -> None:
        assert not MPI_ON, "This method is not available in MPI execution."

        run_script = self.path.joinpath(f"{self.name}.sh")
        assert run_script.exists(), "No script found."

        env = os.environ.copy()
        _ = env.pop("BAMBOOST_MPI", None)  # remove bamboost MPI environment variable
        subprocess.run(["bash", f"{run_script}"], env=env)
        log.info(f'Simulation "{self.name}" submitted.')
        self.metadata["submitted"] = True
