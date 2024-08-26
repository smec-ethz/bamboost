# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

from contextlib import contextmanager
from typing import Literal, TypedDict

import numpy as np

from bamboost.common.file_handler import open_h5file
from bamboost.common.mpi import MPI
from bamboost.simulation_writer import SimulationWriter

try:
    import fenics as fe
except ImportError:
    raise ImportError("FEniCS not found. Module unavailable.")

__all__ = ["FenicsWriter"]


class FenicsWriter(SimulationWriter):
    """
    Helper writer for input from FEniCS directly.

    Args:
        uid: Unique identifier for the simulation
        path: Path to database
        comm: MPI communicator
    """

    def __init__(
        self,
        uid: str,
        path: str,
        comm: MPI.Comm = MPI.COMM_WORLD,
        create_if_not_exists: bool = False,
    ):
        super().__init__(uid, path, comm)

    def add_field(
        self,
        name: str,
        func: fe.Function,
        time: float = None,
        mesh: str = None,
        dtype: str = None,
        center: Literal["Node", "Cell"] = "Node",
    ) -> None:
        """Add a dataset to the file. The data is stored at `data/`.

        Args:
            name: Name for the dataset
            func: FEniCS function to store
            time: Optional. time
            mesh: Optional. Linked mesh for this data
            dtype: Optional. Numpy style datatype, see h5py documentation,
                defaults to the dtype of the vector
            center: Optional. Center of the data. Can be 'Node' or 'Cell'.
                Default is 'Node'.
        """
        mesh = mesh if mesh is not None else self._default_mesh
        time = time if time is not None else self.step

        self._dump_fenics_field(
            f"data/{name}/{self.step}",
            func,
            dtype=dtype,
            center=center,
        )
        self._comm.barrier()  # attempt to fix bug (see SimulationWriter add_field)

        if self._prank == 0:
            with self._file("a"):
                vec = self._file["data"][name][str(self.step)]
                vec.attrs.update({"center": center, "mesh": mesh, "t": time})

        self._comm.barrier()  # attempt to fix bug (see SimulationWriter add_field)

    def _dump_fenics_field(
        self,
        location: str,
        field: fe.Function,
        dtype: str = None,
        center: Literal["Node", "Cell"] = "Node",
    ) -> None:
        # get global dofs ordering and vector
        if center == "Node":
            data = self._get_global_dofs(field)
        elif center == "Cell":
            data = self._get_global_dofs_cell_data(field)
        else:
            raise ValueError("Center must be 'Node' or 'Cell'.")

        vector = data["vector"]
        global_map = data["global_map"]
        global_size = data["global_size"]

        dim = data["vector"].shape[1:] if data["vector"].ndim > 1 else None

        group_name, dataset_name = location.rstrip("/").rsplit("/", 1)

        # Write vector to file
        with self._file("a", driver="mpio", comm=self._comm) as f:
            grp = f.require_group(group_name)
            vec = grp.require_dataset(
                dataset_name,
                shape=(global_size, *dim) if dim else (global_size,),
                dtype=dtype if dtype else vector.dtype,
            )
            vec[global_map] = vector

    class FenicsFieldInformation(TypedDict):
        vector: np.ndarray
        global_map: np.ndarray
        global_size: int

    def _get_global_dofs(self, func: fe.Function) -> FenicsFieldInformation:
        """
        Get global dofs for a given function.

        Args:
            - func: Expression/field/function

        Returns:
            A dict with the local vector, a mapping from the local to the
            global vertices (both sorted by index because h5py complains if
            slicing is not continuously increasing), and the number of global
            vertices.
        """
        assert hasattr(
            func, "function_space"
        ), "Input is likely an indexed coefficient. Project to it's own function space first."

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

        return {
            "vector": local_vector,
            "global_map": global_vertices[sort_indices],
            "global_size": global_size,
        }

    def _get_global_dofs_cell_data(self, func: fe.Function) -> FenicsFieldInformation:
        """
        Get global dofs for a given function.

        Args:
            - func: Expression/field/function

        Returns:
            A dict with the local vector, a mapping from the local to the
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

        return {
            "vector": local_vector,
            "global_map": local_to_global_indices,
            "global_size": self._comm.allreduce(mesh.num_cells(), op=MPI.SUM),
        }

    def add_mesh(self, mesh: fe.Mesh, mesh_name: str = None) -> None:
        """
        Add the mesh to file using fe.HDF5File. I can't figure out how to
        extract the local mesh data in correct order when running in parallel.

        Args:
            mesh: FEniCS mesh object
            mesh_name: name for mesh (default = `mesh`)
        """
        mesh_name = mesh_name if mesh_name is not None else self._default_mesh
        mesh_location = f"{self._mesh_location}/{mesh_name}/"

        assert not self._file.file_object, "File is open -> Quitting"

        @contextmanager
        def temporary_close_file():
            was_open = False
            if self._file.file_object:
                self._file.file_object.close()
                was_open = True
            try:
                yield
            finally:
                if was_open:
                    self._file.file_object = open_h5file(
                        self._file.file_name,
                        self._file.mode,
                        self._file.driver,
                        self._file.comm,
                    )

        with temporary_close_file():
            with fe.HDF5File(self._comm, self.h5file, "a") as f:
                f.write(mesh, mesh_location)
