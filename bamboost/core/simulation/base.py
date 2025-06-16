"""Simulation management module for bamboost.

This module provides classes and utilities for managing simulations,
including reading and writing simulation data, handling metadata and parameters,
interfacing with HDF5 files, managing simulation status, and supporting
HPC job submission and MPI parallelism.

Classes:
    Status: Enum representing the status of a simulation.
    StatusInfo: Dataclass for detailed status information.
    SimulationName: Utility for generating unique simulation names.
    _Simulation: Abstract base class for simulation objects.
    Simulation: Read-only simulation object.
    SimulationWriter: Writable simulation object for editing and managing simulations.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from abc import ABC
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sized, Union

import numpy as np
from typing_extensions import Self

from bamboost import BAMBOOST_LOGGER, config, constants
from bamboost._typing import _MT, Immutable, Mutable
from bamboost.core import utilities
from bamboost.core.hdf5.file import FileMode, H5Object, HDF5File
from bamboost.core.hdf5.ref import Group
from bamboost.core.simulation.dict import Links, Metadata, Parameters
from bamboost.core.simulation.groups import GroupGit, GroupMesh, GroupMeshes
from bamboost.core.simulation.series import Series
from bamboost.index import CollectionUID, Index
from bamboost.index.sqlmodel import SimulationORM
from bamboost.mpi import MPI_ON, Communicator
from bamboost.utilities import StrPath

if TYPE_CHECKING:
    from bamboost.mpi import Comm


log = BAMBOOST_LOGGER.getChild("simulation")


class Status(Enum):
    """Enum representing the status of a simulation.

    Attributes:
        INITIALIZED: The simulation has been initialized but not yet started.
        STARTED: The simulation is currently running.
        FINISHED: The simulation has completed successfully.
        FAILED: The simulation has failed.
        UNKNOWN: The status of the simulation is unknown.
    """

    INITIALIZED = "initialized"
    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"
    UNKNOWN = "unknown"

    def format(self) -> str:
        """Return the string representation of the status."""
        return self.value


@dataclass
class StatusInfo:
    """Detailed status information for a simulation.

    Attributes:
        status (Status): The current status of the simulation.
        message (Optional[str]): An optional message providing additional details about the status.
    """

    status: Status
    message: Optional[str] = None

    @classmethod
    def parse(cls, status: str) -> StatusInfo:
        import re

        pattern = r"^(?P<status>\w+)(?:\s*\[(?P<message>.+)\])?$"
        match = re.match(pattern, status.strip())

        if match:
            status_str = match.group("status").lower()
            message = match.group("message")
            try:
                return cls(Status(status_str), message)
            except ValueError:
                return cls(Status.UNKNOWN, status)
        else:
            return cls(Status.UNKNOWN, status)

    def format(self) -> str:
        return (
            f"{self.status.value} [{self.message}]"
            if self.message
            else self.status.value
        )


class SimulationName(str):
    """
    Utility class for generating unique simulation names.

    The `SimulationName` class provides a convenient way to generate or validate
    unique names for simulation runs. If no name is provided, a random unique
    identifier is generated using UUID, truncated to the specified length.

    Args:
        name (Optional[str]): The desired name for the simulation. If not provided,
            a unique name will be generated.
        length (int): The length of the generated name if `name` is not provided.
            Default is 10.

    Examples:
        >>> sim_name = SimulationName()  # Generates a unique name of length 10
        >>> sim_name = SimulationName("my_simulation")
        >>> print(sim_name)
        my_simulation

    Note:
        The generated name is guaranteed to be unique across MPI ranks by broadcasting
        the generated UUID from the root rank.
    """

    def __new__(cls, name: Optional[str] = None, length: int = 10):
        name = name or cls.generate_name(length)
        return super().__new__(cls, name)

    @staticmethod
    def generate_name(length: int) -> str:
        if Communicator._active_comm.rank == 0:
            uid = uuid.uuid4().hex[:length]
        else:
            uid = ""
        uid: str = Communicator._active_comm.bcast(uid, root=0)
        return uid


class _Simulation(H5Object[_MT], ABC):
    """
    Abstract base class for simulation objects.

    This class should not be instantiated directly. Use `Simulation` for read-only access
    or `SimulationWriter` for writable access to simulation data.

    Args:
        name (str): Name of the simulation.
        parent (StrPath): Path to the parent or collection directory.
        comm (Optional[Comm]): MPI communicator. Defaults to MPI.COMM_WORLD.
        index (Optional[Index]): Index object. Defaults to the global index file.

    Raises:
        FileNotFoundError: If the simulation does not exist at the specified path.
        InvalidCollectionError: If the parent path is no collection or does not exist.
    """

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

        # MPI information
        self._psize: int = self._comm.size
        self._prank: int = self._comm.rank
        self._ranks = np.array([i for i in range(self._psize)])

        # Reference to the database
        # 06.03.2025: Maybe use the default index instance instead of a new one...
        self._index: Index = index or Index(comm=self._comm)

        # Shortcut to collection uid if available, otherwise resolve it
        self.collection_uid: CollectionUID = kwargs.pop(
            "collection_uid", None
        ) or self._index.resolve_uid(self.path.parent)

        self._data_file: Path = self.path.joinpath(constants.HDF_DATA_FILE_NAME)
        self._xdmf_file: Path = self.path.joinpath(constants.XDMF_FILE_NAME)
        self._bash_file: Path = self.path.joinpath(constants.RUN_FILE_NAME)

    def __eq__(self, other: _Simulation, /) -> bool:
        return (
            self.uid == other.uid
            and self.name == other.name
            and self.path == other.path
            and self.mutable == other.mutable
        )

    def _repr_html_(self):
        import pkgutil

        from jinja2 import Template

        metadata = self.metadata
        parameters_filtered = {
            k: "..."
            if isinstance(v, Sized) and not isinstance(v, str) and len(v) > 5
            else v
            for k, v in self.parameters.items()
        }

        def get_pill_div(text: str, color: str) -> str:
            return (
                f'<div class="status" style="background-color:'
                f'var(--bb-{color});">{text}</div>'
            )

        def get_status_pill(status: StatusInfo) -> str:
            if status.status == Status.FAILED:
                return get_pill_div(status.format(), "red")
            elif status.status == Status.FINISHED:
                return get_pill_div(status.format(), "green")
            elif status.status in (Status.INITIALIZED, Status.UNKNOWN):
                return get_pill_div(status.format(), "grey")
            elif status.status == Status.STARTED:
                return get_pill_div(status.format(), "orange")
            else:
                return get_pill_div(status.format(), "grey")

        def get_submitted_pill(submitted: bool) -> str:
            return (
                get_pill_div("Submitted", "green")
                if submitted
                else get_pill_div("Not submitted", "grey")
            )

        html_string = pkgutil.get_data("bamboost", "_repr/simulation.html").decode()
        icon = pkgutil.get_data("bamboost", "_repr/icon.txt").decode()
        template = Template(html_string)
        file_tree = str(self.files).replace("\n", "</br>").replace(" ", "&nbsp;")

        return template.render(
            uid=self.name,
            icon=icon,
            tree=file_tree,
            parameters=parameters_filtered,
            note=metadata.get("description"),
            status=get_status_pill(self.status),
            submitted=get_submitted_pill(metadata.get("submitted", False)),
            timestamp=metadata.get("created_at", "N/A"),
        )

    @cached_property
    def root(self) -> Group[_MT]:
        return Group("/", self._file)

    @property
    def mutable(self) -> bool:
        return self._file.mutable

    @property
    def _orm(self) -> SimulationORM:
        return self._index.simulation(self.collection_uid, self.name)

    @classmethod
    def from_uid(cls, uid: str, **kwargs) -> Self:
        """
        Return the `Simulation` instance corresponding to the given UID.

        Args:
            uid (str): The full simulation UID in the format "<collection_uid>:<simulation_name>".
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            Self: An instance of the simulation class corresponding to the UID.

        Examples:
            >>> sim = Simulation.from_uid("abc123:mysim")
        """
        collection_uid, name = uid.split(constants.UID_SEPARATOR)
        index = kwargs.pop("index", None) or Index.default
        collection_path = index.resolve_path(collection_uid)
        return cls(name, collection_path, index=index, **kwargs)

    @property
    def uid(self) -> str:
        """
        Returns the full unique identifier (UID) of the simulation.

        The UID is constructed as "<collection_uid>:<simulation_name>", where
        `collection_uid` is the unique identifier of the collection containing
        the simulation, and `simulation_name` is the name of the simulation.

        Returns:
            str: The full UID of the simulation in the format "collection_uid:simulation_name".
        """
        return f"{self.collection_uid}{constants.UID_SEPARATOR}{self.name}"

    def edit(self) -> SimulationWriter:
        """
        Return a mutable `SimulationWriter` object for editing the simulation.

        This method provides an interface to obtain a mutable version of the current
        simulation, allowing modifications to simulation data, metadata, and parameters.

        Returns:
            SimulationWriter: An object with write access to the simulation.

        Examples:
            >>> with sim.edit() as sim_writer:
            ...     sim_writer.parameters["new_param"] = 42
        """
        return SimulationWriter(
            self.name,
            self.path.parent,
            self._comm,
            self._index,
            collection_uid=self.collection_uid,
        )

    def update_database(
        self,
        *,
        metadata: Optional[Mapping] = None,
        parameters: Optional[Mapping] = None,
    ) -> None:
        """Push update to sqlite database.

        Args:
            metadata: metadata dictionary to insert
            parameters: parameter dictionary to insert
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
        """
        Returns the parameters associated with this simulation.

        Returns:
            Parameters[_MT]: The parameters object for this simulation.
        """
        return Parameters(self)

    @cached_property
    def metadata(self) -> Metadata[_MT]:
        """
        Returns the metadata associated with this simulation.

        Returns:
            Metadata[_MT]: The metadata object for this simulation.
        """
        return Metadata(self)

    @property
    def status(self) -> StatusInfo:
        """
        Returns the current status of the simulation.

        Returns:
            StatusInfo: The status information for this simulation.
        """
        try:
            return StatusInfo.parse(self.metadata.__getitem__("status"))
        except KeyError:
            return StatusInfo(Status.UNKNOWN)

    @property
    def created_at(self) -> datetime:
        """
        Returns the creation timestamp of the simulation.

        Returns:
            datetime: The datetime when the simulation was created.
        """
        return self.metadata.__getitem__("created_at")

    @property
    def description(self) -> str:
        """
        Returns the description of the simulation.

        Returns:
            str: The description string.
        """
        return self.metadata.__getitem__("description")

    @cached_property
    def links(self) -> Links[_MT]:
        """
        Returns the links associated with this simulation.

        Returns:
            Links[_MT]: The links object for this simulation.
        """
        return Links(self)

    @cached_property
    def files(self):
        """
        Returns a file picker utility for the simulation directory.

        Returns:
            FilePicker: Utility for browsing files in the simulation directory.
        """
        return utilities.FilePicker(self.path)

    @cached_property
    def git(self) -> GroupGit[_MT]:
        """
        Returns the Git group associated with this simulation.

        Returns:
            GroupGit[_MT]: The Git group object for this simulation.
        """
        return GroupGit(self)

    @cached_property
    def data(self) -> Series[_MT]:
        """
        Returns the default data series for this simulation.

        Returns:
            Series[_MT]: The default data series object.
        """
        return Series(self, path=constants.PATH_DATA)

    @cached_property
    def meshes(self) -> GroupMeshes[_MT]:
        return GroupMeshes(self)

    @cached_property
    def mesh(self) -> GroupMesh:
        return GroupMesh(self, constants.DEFAULT_MESH_NAME)

    def open_in_paraview(self) -> None:
        """Open the xdmf file in paraview."""
        subprocess.call(["paraview", self._xdmf_file])

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

    def require_series(self, path: str) -> Series[_MT]:
        """
        Return a Series object for the given path.

        A "series" in bamboost is a logical group in the HDF5 file that stores
        time-dependent or indexed simulation data, such as fields, scalars, or
        other arrays. Each series is identified by its path and can contain
        multiple fields and steps.

        Args:
            path: Path to the series group within the simulation HDF5 file.

        Returns:
            Series[_MT]: The Series object for the specified path.
        """
        return Series(self, path=path)

    def create_xdmf(
        self,
        field_names: Optional[Iterable[str]] = None,
        timesteps: Optional[Iterable[float]] = None,
        *,
        series: Optional[Series[_MT]] = None,
        filename: Optional[StrPath] = None,
        mesh_name: str = constants.DEFAULT_MESH_NAME,
    ):
        """
        Generate an XDMF file for visualization of simulation data.

        This method creates an XDMF file that references the simulation's mesh and
        time-dependent field data, enabling visualization in tools such as ParaView.

        Args:
            field_names (Optional[Iterable[str]]): Names of the fields to include in the XDMF file.
                If None, all available fields in the series are included.
            timesteps (Optional[Iterable[float]]): List of timesteps to include.
                If None, all timesteps in the series are included.
            series (Optional[Series[_MT]]): The data series to use for field and timestep information.
                If None, uses the default data series (`self.data`).
            filename (Optional[StrPath]): Path to the output XDMF file.
                If None, defaults to "<simulation_path>/data.xdmf".
            mesh_name (str): Name of the mesh to reference in the XDMF file.
                Defaults to the constant `DEFAULT_MESH_NAME`.

        Examples:
            >>> sim.create_xdmf(field_names=["velocity", "pressure"])
            >>> sim.create_xdmf(timesteps=[0.0, 0.1, 0.2], filename="custom.xdmf")
        """
        from bamboost.core.simulation.xdmf import XDMFWriter

        series = series or self.data
        fields = series.get_fields(*field_names if field_names else [])
        filename = filename or self.path.joinpath(constants.XDMF_FILE_NAME)
        timesteps = timesteps if timesteps is not None else series.values

        def _create_xdmf():
            xdmf = XDMFWriter(self._file)
            xdmf.add_mesh(self.meshes[mesh_name])
            xdmf.add_timeseries(timesteps, fields, mesh_name)
            xdmf.write_file(filename)
            log.debug(f"produced XDMF file at {filename}")

        self.post_write_instruction(_create_xdmf)


class Simulation(_Simulation[Immutable]):
    """
    Read-only simulation object.

    The `Simulation` class provides read-only access to simulation data, metadata,
    and parameters. It is intended for inspecting and analyzing existing simulations
    without modifying their contents. For editing or managing simulations, use
    `SimulationWriter` (or `sim_writer = sim.edit()`).

    Args:
        name (str): Name of the simulation.
        parent (StrPath): Path to the parent or collection directory.
        comm (Optional[Comm]): MPI communicator. Defaults to MPI.COMM_WORLD.
        index (Optional[Index]): Index object. Defaults to the global index file.
        **kwargs: Additional keyword arguments.

    Examples:
        >>> sim = Simulation("mysim", "/path/to/collection")
        >>> print(sim.parameters)
        >>> print(sim.metadata)
    """

    def __init__(
        self,
        name: str,
        parent: StrPath,
        comm: Optional[Comm] = None,
        index: Optional[Index] = None,
        **kwargs,
    ):
        super().__init__(name, parent, comm, index, **kwargs)
        self._file: HDF5File[Immutable] = HDF5File(
            self._data_file, comm=self._comm, mutable=False
        )


class SimulationWriter(_Simulation[Mutable]):
    """
    Mutable simulation object for editing and managing simulations.

    The `SimulationWriter` class provides write access to simulation data, metadata,
    and parameters. It is intended for creating, editing, and managing simulations.
    Use this class when you need to modify the contents of a simulation, such as
    updating parameters, metadata, or simulation data.

    Args:
        name (str): Name of the simulation.
        parent (StrPath): Path to the parent or collection directory.
        comm (Optional[Comm]): MPI communicator. Defaults to MPI.COMM_WORLD.
        index (Optional[Index]): Index object. Defaults to the global index file.
        **kwargs: Additional keyword arguments.

    Examples:
        >>> with SimulationWriter("mysim", "/path/to/collection") as sim_writer:
        ...     sim_writer.parameters["new_param"] = 42
        ...     sim_writer.metadata["description"] = "Updated simulation"
    """

    def __init__(
        self,
        name: str,
        parent: StrPath,
        comm: Optional[Comm] = None,
        index: Optional[Index] = None,
        **kwargs,
    ):
        super().__init__(name, parent, comm, index, **kwargs)
        self._file: HDF5File[Mutable] = HDF5File(
            self._data_file, comm=self._comm, mutable=True
        )

    def __enter__(self) -> SimulationWriter:
        self.status = Status.STARTED
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.status = StatusInfo(Status.FAILED, str(exc_val))
            log.error(
                f"Simulation failed with {exc_type.__name__}: {exc_val}\nTraceback: {exc_tb}"
            )
            return
        self.status = Status.FINISHED

    def initialize(self) -> None:
        """Initialize the simulation.

        This method sets up the simulation's HDF5 data file and required groups,
        and records initial metadata such as status and creation time.
        """
        # create the data file
        with self._file.open(FileMode.APPEND, driver="mpio") as f:
            self.metadata.update(
                {
                    "status": Status.INITIALIZED.value,
                    "created_at": datetime.now(),
                }
            )
            # create groups
            f.create_group(constants.PATH_PARAMETERS)
            f.create_group(constants.PATH_LINKS)
            f.create_group(constants.PATH_MESH)

            # create default series ('data')
            self._initialize_series(constants.PATH_DATA)

    def _initialize_series(self, path: str) -> None:
        """Create the groups for a series. Does not manage file state.

        Args:
            path: path of the series
        """
        # add series to metadata for easier retrieval
        all_series = set(self.metadata.get(".series_paths", []))
        all_series.add(str(path))
        self.metadata.set(".series_paths", list(all_series))

        f = self._file
        grp = f.require_group(path)
        grp.attrs[".series"] = True
        grp.require_group(constants.RELATIVE_PATH_FIELD_DATA)
        grp.require_group(constants.RELATIVE_PATH_SCALAR_DATA)

    @_Simulation.status.setter
    def status(self, value: Union[StatusInfo, Status]) -> None:
        self.metadata.__setitem__("status", value.format())

    @_Simulation.description.setter
    def description(self, value: str) -> None:
        self.metadata.__setitem__("description", value)

    def require_series(self, path: str) -> Series[Mutable]:
        # require the group in the HDF5 file
        with self._file.open(FileMode.APPEND, driver="mpio"):
            if path not in self.root.keys():
                self._initialize_series(path)
        return super().require_series(path)

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
        euler: bool = False,
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
            f"export SIMULATION_DIR={self.path.as_posix()}\n"
            f"export SIMULATION_ID={self.uid}\n\n"
        )
        script += "\n".join(commands)

        with self._bash_file.open("w") as file:
            file.write(script)

        self.metadata["submitted"] = False

    def run_simulation(self, executable: str = "bash") -> None:
        """
        Run the simulation using the specified executable and the default run file in
        the simulation directory.

        This method executes the simulation's run script using the provided executable.
        By default, it uses "bash" to run the script locally. If "sbatch" is specified,
        the script will be submitted to a Slurm workload manager.

        Args:
            executable (str): The executable to use for running the simulation script.
                Defaults to "bash". Use "sbatch" to submit to a Slurm scheduler.

        Raises:
            AssertionError: If called during MPI execution.
            FileNotFoundError: If the run script does not exist.

        Examples:
            >>> sim_writer.run_simulation()  # Runs locally with bash
            >>> sim_writer.run_simulation(executable="sbatch")  # Submits to Slurm
        """
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
        """
        Submit the simulation to a job scheduler.

        This method submits the simulation's run script to a job scheduler using "sbatch".
        It is a convenience wrapper around `run_simulation(executable="sbatch")`.

        Examples:
            >>> sim_writer.submit_simulation()
        """
        self.run_simulation(executable="sbatch")
