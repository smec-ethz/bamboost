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
from abc import ABC
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Optional,
    Sized,
    TypeAlias,
    overload,
)

import numpy as np
from simmr import Simulation as _Simulation_simmr
from simmr import Status as _Status_simmr
from typing_extensions import Self

from bamboost import constants, utilities
from bamboost._logger import BAMBOOST_LOGGER
from bamboost._typing import _MT, Immutable, Mutable
from bamboost.hdf5.file import FileMode, H5Object, HDF5File
from bamboost.hdf5.ref import Group
from bamboost.mpi import ReuseComm
from bamboost.simulation.groups import GroupGit, GroupMesh, GroupMeshes
from bamboost.simulation.series import Series
from bamboost.utilities import StrPath

if TYPE_CHECKING:
    from simmr._core import SimulationMetadata

    from bamboost.collection import Collection
    from bamboost.mpi import Comm

    cached_property: TypeAlias = property  # noqa: PYI042


log = BAMBOOST_LOGGER.getChild("simulation")


@dataclass
class Status:
    """Enum representing the status of a simulation. Includes the optional message for
    more details."""

    state: _Status_simmr
    msg: str | None = None

    @classmethod
    def with_msg(cls, status: _Status_simmr, msg: str | None = None) -> Self:
        """Create a Status with an optional message."""
        return cls(status, msg)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Status):
            return self.state == other.state and self.msg == other.msg
        if isinstance(other, _Status_simmr):
            return self.state == other
        return NotImplemented


@dataclass(frozen=True, init=False)  # init=False because we handle it in __new__
class SimulationUID:
    """UID of a simulation, consisting of the collection UID and the simulation name.

    Use `str(SimulationUID(...))` to get the string representation of the UID, which is in
    the format `<collection_uid>:<simulation_name>`. The constructor can be called with
    either the string representation or the collection UID and simulation name as separate
    arguments.
    """

    collection_uid: str
    simulation_name: str

    @overload
    def __new__(cls, uid: str | SimulationUID, /) -> Self: ...

    @overload
    def __new__(cls, collection_uid: str, simulation_name: str, /) -> Self: ...

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], SimulationUID):
            return args[0]

        instance = super().__new__(cls)

        if len(args) == 1 and isinstance(args[0], str):
            try:
                c_uid, s_name = args[0].split(constants.UID_SEPARATOR, 1)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid SimulationUID string: {args[0]!r}. expected format "
                    f"'<collection_uid>{constants.UID_SEPARATOR}<simulation_name>'"
                ) from exc
            object.__setattr__(instance, "collection_uid", str(c_uid))
            object.__setattr__(instance, "simulation_name", str(s_name))

        elif len(args) == 2:
            object.__setattr__(instance, "collection_uid", str(args[0]))
            object.__setattr__(instance, "simulation_name", str(args[1]))

        else:
            raise ValueError("Invalid arguments for SimulationUID")

        return instance

    @classmethod
    def from_uri(cls, uri: str) -> Self:
        """Create a SimulationUID from a babo URI.

        Args:
            uri: The URI to parse. Expected format is
            `sim://<collection_uid>/<simulation_name>`.
        """
        if not uri.startswith("sim://"):
            raise ValueError(
                f"Invalid URI: {uri}. Expected format is sim://<collection_uid>/<simulation_name>"
            )
        _, rest = uri.split("://", 1)
        uid, name = rest.split("/", 1)
        return cls(uid, name)

    def to_uri(self) -> str:
        """Return the babo URI representation of the SimulationUID.

        Returns:
            str: The URI in the format `sim://<collection_uid>/<simulation_name>`.
        """
        return f"sim://{self.collection_uid}/{self.simulation_name}"

    def __eq__(self, other):
        if isinstance(other, SimulationUID):
            return (
                self.collection_uid == other.collection_uid
                and self.simulation_name == other.simulation_name
            )
        if isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __str__(self):
        return f"{self.collection_uid}{constants.UID_SEPARATOR}{self.simulation_name}"

    def __getnewargs__(self):
        # This method is used by pickle to serialize the object.
        # It's needed to communicate the object across MPI ranks with bcast.
        return (str(self),)


class Links(Mapping[str, "Simulation"]):
    """Links associated with a simulation.

    This class provides a mapping interface to access links between simulations.
    Each link is represented by a key-value pair, where the key is a string and
    the value is a `SimulationUID` object.

    Args:
        simulation (_Simulation): The simulation object to which the links belong.
    """

    def __init__(self, simulation: _Simulation[_MT]) -> None:
        self._simulation = simulation
        self._dict: dict[str, SimulationUID] = {
            key: SimulationUID.from_uri(value)
            for key, value in simulation.metadata.get("links", {}).items()
        }

    def __str__(self) -> str:
        return self._dict.__str__()

    __repr__ = __str__

    def __getitem__(self, key: str) -> Simulation:
        uid = self._dict[key]
        return Simulation.from_uid(uid)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def update(self, new_links: dict[str, SimulationUID]) -> None:
        """Update the links in the simulation metadata.

        Args:
            new_links (dict[str, str]): A dictionary of new links to add or update.
        """
        merged = {**self._dict, **new_links}
        as_uri = {k: v.to_uri() for k, v in merged.items()}
        self._simulation._core.update_links(as_uri)


class _Simulation(H5Object[_MT], ABC):
    """
    Abstract base class for simulation objects.

    This class should not be instantiated directly. Use `Simulation` for read-only access
    or `SimulationWriter` for writable access to simulation data.

    Args:
        name (str): Name of the simulation.
        parent (StrPath): Path to the parent or collection directory.
        comm (Optional[Comm]): MPI communicator. Defaults to MPI.COMM_WORLD.

    Raises:
        FileNotFoundError: If the simulation does not exist at the specified path.
        InvalidCollectionError: If the parent path is no collection or does not exist.
    """

    collection_uid: str

    def __init__(
        self,
        name: str,
        parent: StrPath,
        comm: Comm | ReuseComm | None = None,
        collection: Collection | None = None,
    ):
        from bamboost.collection import Collection

        self.name: str = name
        self.path: Path = Path(parent).joinpath(name).absolute()
        self._collection = collection or Collection(parent)
        self.collection_uid = self._collection.uid
        self._core = _Simulation_simmr.from_path(self.path.as_posix())

        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Simulation {self.name} does not exist in {self.path}."
            )

        if comm is not None:
            self._comm = comm

        self._psize: int = self._comm.size
        self._prank: int = self._comm.rank
        self._ranks = np.array([i for i in range(self._psize)])

        self._data_file: Path = self.path.joinpath(constants.HDF_DATA_FILE_NAME)

    @classmethod
    def from_core(
        cls, core: _Simulation_simmr, *, comm: Comm | ReuseComm | None = None
    ) -> Self:
        """Create a Simulation instance from a core simulation object."""
        from bamboost.collection import Collection

        collection = Collection(core.metadata["collection_uid"])

        instance = cls.__new__(cls)
        instance._collection = collection
        instance._core = core
        instance.name = core.name
        instance.path = collection.path.joinpath(core.name)
        instance.collection_uid = collection.uid

        if comm is not None:
            instance._comm = comm

        return instance

    @classmethod
    def from_uid(
        cls, uid: str | SimulationUID, *, comm: Comm | ReuseComm | None = None
    ) -> Self:
        """Return the `Simulation` instance corresponding to the given UID.

        Args:
            uid: The full simulation UID in the format "<collection_uid>:<simulation_name>".
            comm: Optional MPI communicator to use for the simulation instance.

        Returns:
            Self: An instance of the simulation class corresponding to the UID.

        Examples:
            >>> sim = Simulation.from_uid("abc123:mysim")
        """
        from simmr._core import Collection as _Collection_rust

        uid = SimulationUID(uid)
        collection_uid, name = uid.collection_uid, uid.simulation_name
        collection = _Collection_rust(collection_uid)
        sim_core = _Simulation_simmr.from_collection(collection, name)
        return cls.from_core(sim_core, comm=comm)

    @classmethod
    def from_cwd(cls, *, comm: Comm | ReuseComm | None = None) -> Self:
        """Return the `Simulation` instance located in the current working directory.

        Args:
            comm: Optional MPI communicator to use for the simulation instance.

        Raises:
            FileNotFoundError: If the current working directory is not a valid simulation path.

        Examples:
            >>> sim = Simulation.from_cwd()
        """
        sim_core = _Simulation_simmr.from_cwd()
        return cls.from_core(sim_core, comm=comm)

    @property
    def file(self) -> HDF5File[_MT]:
        if hasattr(self, "_file"):
            return self._file
        raise AttributeError(
            "Simulation HDF file is not initialized. If you never used it, you "
            "must use SimulationWriter to get a mutable object."
        )

    @file.setter
    def file(self, value: HDF5File[_MT]) -> None:
        self._file = value

    def __eq__(self, other: _Simulation, /) -> bool:  # ty:ignore[invalid-method-override]
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

        def get_status_pill(status: Status) -> str:
            if status == _Status_simmr.Failed:
                return get_pill_div(str(status.state), "red")
            elif status == _Status_simmr.Completed:
                return get_pill_div(str(status.state), "green")
            elif status in (_Status_simmr.Unknown, _Status_simmr.Initialized):
                return get_pill_div(str(status.state), "grey")
            elif status == _Status_simmr.Running:
                return get_pill_div(str(status.state), "orange")
            else:
                return get_pill_div(str(status.state), "grey")

        def get_submitted_pill(submitted: bool) -> str:
            return (
                get_pill_div("Submitted", "green")
                if submitted
                else get_pill_div("Not submitted", "grey")
            )

        html_string = pkgutil.get_data("bamboost", "_repr/simulation.html")
        icon = pkgutil.get_data("bamboost", "_repr/icon.txt")
        assert html_string is not None and icon is not None, (
            "Failed to load HTML template or icon."
        )
        icon = icon.decode()
        html_string = html_string.decode()
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
        return Group("/", self.file)

    @property
    def mutable(self) -> bool:
        return self.file.mutable

    @property
    def uid(self) -> SimulationUID:
        """
        Returns the unique identifier (UID) of the simulation.

        The UID is constructed as "<collection_uid>:<simulation_name>", where
        `collection_uid` is the unique identifier of the collection containing
        the simulation, and `simulation_name` is the name of the simulation.

        Returns:
            UID of the simulation. To get the UID in the format
            "collection_uid:simulation_name" use `str(uid)`.
        """
        return SimulationUID(self.collection_uid, self.name)

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
            ReuseComm(self),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Returns the parameters associated with this simulation.

        Returns:
            Parameters[_MT]: The parameters object for this simulation.
        """
        return self._core.parameters

    @property
    def metadata(self) -> SimulationMetadata:
        """
        Returns the metadata associated with this simulation.

        Returns:
            Metadata[_MT]: The metadata object for this simulation.
        """
        return self._core.metadata

    @property
    def status(self) -> Status:
        """
        Returns the current status of the simulation.

        Returns:
            StatusInfo: The status information for this simulation.
        """
        metadata = self._core.metadata
        status = metadata.get("status")
        msg = metadata.get("status_message")
        return Status.with_msg(status, msg)

    @property
    def status_message(self) -> str | None:
        """
        Returns a human-readable message describing the current status of the simulation.

        Returns:
            str: A message describing the simulation's status.
        """
        return self._core.metadata.get("status_message")

    @property
    def links(self) -> Links:
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

    @property
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

    def update_status(self, status: Status) -> None:
        """Update the status of the simulation.

        Args:
            status: The new status to set for the simulation.
        """
        self._core.update_status(status.state, status.msg)

    def run(self, stage: str) -> None:
        return self._core.run(stage)

    def submit(self, stage: str) -> str | None:
        return self._core.submit(stage)

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
        from bamboost.simulation.xdmf import XDMFWriter

        series = series or self.data
        fields = series.get_fields(*field_names if field_names else [])
        filename = filename or self.path.joinpath(constants.XDMF_FILE_NAME)
        timesteps = timesteps if timesteps is not None else series.values

        def _create_xdmf():
            xdmf = XDMFWriter(self.file)
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
        comm: Comm | ReuseComm | None = None,
    ):
        super().__init__(name, parent, comm)
        try:
            self.file = HDF5File(self._data_file, comm=ReuseComm(self), mutable=False)
        except FileNotFoundError:
            pass


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
        comm: Comm | ReuseComm | None = None,
    ):
        super().__init__(name, parent, comm)
        self.file = HDF5File(
            self._data_file, comm=ReuseComm(self), mutable=True
        )._create_file()

    # TODO: decide whether this is desirable
    def __enter__(self) -> Self:
        self.update_status(Status(_Status_simmr.Running))
        return self

    # TODO: decide whether this is desirable
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._core.update_status(_Status_simmr.Failed, str(exc_val))
            log.error(
                f"Simulation failed with {exc_type.__name__}: {exc_val}\nTraceback: {exc_tb}"
            )
            return

        self.update_status(Status(_Status_simmr.Completed))

    def require_series(self, path: str) -> Series[Mutable]:
        # require the group in the HDF5 file
        with self.file.open(FileMode.APPEND, driver="mpio"):
            if path not in self.root.keys():  # noqa: SIM118
                self._initialize_series(path)
        return super().require_series(path)

    @property
    def data(self) -> Series[Mutable]:
        """Returns the default data series for this simulation.

        Returns:
            Series[Mutable]: The default data series object.
        """
        with self.file.open(FileMode.READ, driver="mpio"):
            if constants.PATH_DATA not in self.root.keys():  # noqa: SIM118
                self._initialize_series(constants.PATH_DATA)
        return Series(self, path=constants.PATH_DATA)

    def _initialize_series(self, path: str) -> None:
        """Create the groups for a series. Does not manage file state.

        Args:
            path: path of the series
        """
        # add series to metadata for easier retrieval
        root_attrs = self.root.attrs
        all_series = set(root_attrs.get(".series_paths", []))
        all_series.add(str(path))
        root_attrs.set(".series_paths", list(all_series))

        f = self.file
        grp = f.require_group(path)
        grp.attrs[".series"] = True
        grp.require_group(constants.RELATIVE_PATH_FIELD_DATA)
        grp.require_group(constants.RELATIVE_PATH_SCALAR_DATA)

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
