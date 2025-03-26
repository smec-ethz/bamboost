from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import h5py
import numpy as np

from bamboost import BAMBOOST_LOGGER
from bamboost._typing import Mutable
from bamboost.constants import DEFAULT_MESH_NAME
from bamboost.core.hdf5.file import FileMode
from bamboost.core.simulation import CellType, FieldType
from bamboost.core.simulation.groups import GroupMeshes
from bamboost.core.simulation.series import FieldData, Series, StepWriter
from bamboost.mpi import MPI
from bamboost.plugins import Plugin, PluginComponent

try:
    import fenics as fe
except ImportError:
    raise ImportError("FEniCS not found. Module unavailable.")

__all__ = ["FenicsBamboostPlugin"]

log = BAMBOOST_LOGGER.getChild("bamboost_plugin_fenics")

if TYPE_CHECKING:
    from functools import cached_property

    from bamboost.core.simulation import SimulationWriter
    from bamboost.core.simulation.series import Series

    class _T_Simulation(SimulationWriter):
        @cached_property
        def data(self) -> _T_Series: ...
        @cached_property
        def meshes(self) -> FenicsBamboostPlugin.FenicsMeshes: ...

    class _T_Series(Series[Mutable]):
        def require_step(
            self: _T_Series, value: float = ..., step: Optional[int] = ...
        ) -> FenicsBamboostPlugin.FenicsWriter: ...


class WriteStrategy(Enum):
    """Enum for write style."""

    SCATTERED = 0
    """Scattered write style. Each process writes its own data. This is slower because the
    location to write to is not necessarily contiguous."""

    CONTIGUOUS = 1
    """Contiguous write style. Data is gathered on the root process and written
    contigously."""


class _FenicsStepWriter(StepWriter, PluginComponent, replace_base=True):
    """
    Helper writer for input from FEniCS directly.

    Args:
        uid: Unique identifier for the simulation
        path: Path to database
        comm: MPI communicator
        create_if_not_exists: Create file if it does not exist
        write_strategy: Write strategy for the data. Contiguous is faster but
            requires the entire array to fit in memory of the root process.
    """

    def __init__(self, series: Series[Mutable], step: int):
        super().__init__(series, step)

    def add_field(
        self,
        name: str,
        function: fe.Function,
        *,
        mesh_name: str = DEFAULT_MESH_NAME,
        field_type: FieldType = FieldType.NODE,
    ) -> None:
        """Add a Fenics function to the step.

        Args:
            name: The name of the field.
            function: The Fenics function to add.
            mesh_name: (Optional) The name of the mesh to which the field belongs. ()
            field_type: (Optional) The type of the field (default: FieldType.NODE). This
                is only relevant for XDMF writing.
        """
        # field = self._series.get_field(name)
        field = _FenicsFieldData(self._series, name)
        field.require_self()
        field.add_fenics_function(
            str(self._step),
            function,
            file_map=True,
            field_type=field_type,
            attrs={"mesh": mesh_name, "field_type": field_type.value},
        )
        log.debug(f"Added field {name} for step {self._step}")


class _FenicsFieldData(FieldData[Mutable], PluginComponent):
    def add_fenics_function(
        self,
        name: str,
        function: fe.Function,
        field_type: FieldType = FieldType.NODE,
        attrs: Optional[dict[str, Any]] = None,
        dtype: Optional[str] = None,
        *,
        file_map: bool = True,
    ) -> None:
        """Add a Fenics function to the field."""
        write_strategy: WriteStrategy = self.__plugin__.opts["write_strategy"]

        if write_strategy == WriteStrategy.CONTIGUOUS:
            self._dump_fenics_field_on_root(
                name, function, dtype=dtype, center=field_type
            )
        else:
            self._dump_fenics_field(name, function, dtype=dtype, center=field_type)

        log.info(f'Written dataset to "{self._path}/{name}"')

        self.attrs.update(attrs or {})

        # update file_map
        if file_map:
            self._group_map[name] = h5py.Dataset

    def _dump_fenics_field(
        self,
        name: str,
        field: fe.Function,
        dtype: Optional[str] = None,
        center: FieldType = FieldType.NODE,
    ) -> None:
        # get global dofs ordering and vector
        if center == FieldType.NODE:
            vector, global_map, global_size = self._get_global_dofs(field)
        elif center == FieldType.ELEMENT:
            vector, global_map, global_size = self._get_global_dofs_cell_data(field)
        else:
            raise ValueError("Invalid field type (NODE or ELEMENT).")

        dim = vector.shape[1:] if vector.ndim > 1 else None

        with self._file.open(FileMode.APPEND, driver="mpio"):
            dataset = self._obj.require_dataset(
                name,
                shape=(global_size, *dim) if dim else (global_size,),
                dtype=dtype if dtype else vector.dtype,
            )
            dataset[global_map] = vector

    def _dump_fenics_field_on_root(
        self,
        name: str,
        field: fe.Function,
        dtype: Optional[str] = None,
        center: FieldType = FieldType.NODE,
    ) -> None:
        """Assembles the vector on the root process and writes it to file contiguously.

        This is faster but requires the entire array to fit in memory.

        Args:
            location: Location in the HDF5 file
            field: FEniCS function
            dtype: Optional. Numpy style datatype, see h5py documentation,
                defaults to the dtype of the vector
            center: Optional. Center of the data. Can be 'Node' or 'Cell'.
                Default is 'Node'.
        """
        # get global dofs ordering and vector
        if center == FieldType.NODE:
            vector, global_map, global_size = self._get_global_dofs(field)
        elif center == FieldType.ELEMENT:
            vector, global_map, global_size = self._get_global_dofs_cell_data(field)
        else:
            raise ValueError("Invalid field type (NODE or ELEMENT).")

        dim = vector.shape[1:] if vector.ndim > 1 else None

        vector_p = self._file._comm.gather(vector)
        global_map_p = self._file._comm.gather(global_map)

        # On RAM, construct a contiguous vector on the root process
        if self._file._comm.rank == 0:
            vector_contiguous = np.zeros(
                (global_size, *dim) if dim else (global_size,), dtype=vector.dtype
            )
            for map, vec in zip(global_map_p, vector_p):  # type: ignore
                vector_contiguous[map] = vec

            # Write vector to file
            def _write_vector():
                vec = self.require_dataset(
                    name,
                    shape=(global_size, *dim) if dim else (global_size,),
                    dtype=dtype if dtype else vector.dtype,
                )
                vec[:] = vector_contiguous

            self.post_write_instruction(_write_vector)

    def _get_global_dofs(self, func: fe.Function) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Get global dofs for a given function.

        Args:
            - func: Expression/field/function

        Returns:
            A tuple with the local vector, a mapping from the local to the
            global vertices (both sorted by index because h5py complains if
            slicing is not continuously increasing), and the number of global
            vertices.
        """
        assert hasattr(func, "function_space"), (
            "Input is likely an indexed coefficient. Project to it's own function space first."
        )

        # Project to CG1 if necessary
        if func.ufl_element().degree() != 1:
            func = fe.project(
                func, fe.FunctionSpace(func.function_space().mesh(), "CG", 1)
            )

        mesh = func.function_space().mesh()
        global_size = mesh.num_entities_global(0)
        shape = (-1, *func.ufl_shape) if func.ufl_shape else (-1,)
        val_per_vertex = np.prod(shape[1:]).astype(np.int32) if func.ufl_shape else 1

        dofmap = func.function_space().dofmap()
        d2v = fe.dof_to_vertex_map(func.function_space())
        d2v = (
            d2v[np.arange(0, len(d2v), val_per_vertex, dtype=np.int32)]
            // val_per_vertex
        )

        loc0, loc1 = (i // val_per_vertex for i in dofmap.ownership_range())
        global_vertex_numbers = mesh.topology().global_indices(0)
        global_vertices = global_vertex_numbers[d2v[: loc1 - loc0]]
        sort_indices = np.argsort(global_vertices)

        local_vector = (
            func.vector().get_local().reshape(shape)[: loc1 - loc0][sort_indices]
        )

        return (local_vector, global_vertices[sort_indices], global_size)

    def _get_global_dofs_cell_data(
        self, func: fe.Function
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Get global dofs for a given function.

        Args:
            - func: Expression/field/function

        Returns:
            A tuple with the local vector, a mapping from the local to the
            global vertices (both sorted by index because h5py complains if
            slicing is not continuously increasing), and the number of global
            vertices.
        """
        V = func.function_space()
        val_per_vertex = (
            np.prod(func.ufl_shape).astype(np.int32) if func.ufl_shape else 1
        )

        mesh = V.mesh()
        dofmap = V.dofmap()

        local_to_global_indices = dofmap.tabulate_local_to_global_dofs()
        local_to_global_indices = (
            local_to_global_indices[
                np.arange(
                    0, len(local_to_global_indices), val_per_vertex, dtype=np.int32
                )
            ]
            // val_per_vertex
        )
        shape = (-1, *func.ufl_shape) if func.ufl_shape else (-1,)
        local_vector = func.vector().get_local().reshape(shape)

        return (
            local_vector,
            local_to_global_indices,
            self._file._comm.allreduce(mesh.num_cells(), op=MPI.SUM),
        )


class _FenicsMeshes(GroupMeshes[Mutable], PluginComponent, replace_base=True):
    def add_fenics_mesh(
        self,
        mesh: fe.Mesh,
        name: str = DEFAULT_MESH_NAME,
        cell_type: CellType = CellType.TRIANGLE,
    ) -> None:
        """Add a mesh with the given name to the simulation.

        Args:
            nodes: Node coordinates
            cells: Cell connectivity
            name: Name of the mesh
            cell_type: Cell type (default: "triangle"). In general, we do not care about
                the cell type and leave it up to the user to make sense of the data they
                provide. However, the cell type specified is needed for writing an XDMF
                file. For possible types, consult the XDMF/paraview manual.
        """
        from mpi4py import MPI

        mesh_location = f"{self._path}/{name}"
        with self._temporary_close_file():
            with fe.HDF5File(MPI.COMM_WORLD, self._file._filename, "a") as f:
                f.write(mesh, mesh_location)

        self.attrs.update({"cell_type": cell_type.value})

    @contextmanager
    def _temporary_close_file(self):
        was_open = False
        if self._file.is_open:
            was_open = True
            mode = self._file.mode
            driver = str(self._file.driver)
            self._file.close()
        try:
            yield
        finally:
            if was_open:
                self._file.open(mode, driver)  # type: ignore


class _T_FenicsPluginOpts(TypedDict, total=True):
    write_strategy: WriteStrategy
    """Write strategy for the data. Contiguous is faster but requires the entire array to fit in memory of the root process."""


class FenicsBamboostPlugin(Plugin[_T_FenicsPluginOpts]):
    """Plugin for writing FEniCS data to HDF5 files."""

    FenicsWriter = _FenicsStepWriter
    FenicsFieldData = _FenicsFieldData
    FenicsMeshes = _FenicsMeshes
