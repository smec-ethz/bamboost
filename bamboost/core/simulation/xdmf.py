# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, cast

import h5py
import numpy as np

from bamboost._typing import StrPath
from bamboost.core.hdf5.file import FileMode, HDF5File, HDF5Path
from bamboost.core.simulation import FieldType
from bamboost.mpi import Communicator
from bamboost.mpi.utilities import RootProcessMeta

if TYPE_CHECKING:
    from bamboost.core.simulation.groups import GroupMesh
    from bamboost.core.simulation.series import FieldData

__all__ = ["XDMFWriter"]

# `numpy_to_xdmf_dtype` from `meshio.xdmf.common`
numpy_to_xdmf_dtype = {
    "int8": ("Int", "1"),
    "int16": ("Int", "2"),
    "int32": ("Int", "4"),
    "int64": ("Int", "8"),
    "uint8": ("UInt", "1"),
    "uint16": ("UInt", "2"),
    "uint32": ("UInt", "4"),
    "uint64": ("UInt", "8"),
    "float32": ("Float", "4"),
    "float64": ("Float", "8"),
}


class XDMFWriter(metaclass=RootProcessMeta):
    """Write xdmf file for a subset of the stored data in the H5 file.

    Args:
        filename (str): xdmf file path
        h5file (str): h5 file path
    """

    _comm = Communicator()

    def __init__(self, file: HDF5File):
        self._file = file
        self.root_element = ET.Element("Xdmf", Version="3.0")
        self.domain = ET.SubElement(self.root_element, "Domain")
        ET.register_namespace("xi", "https://www.w3.org/2001/XInclude/")

    def write_file(self, filename: StrPath):
        filename = Path(filename)
        tree = ET.ElementTree(self.root_element)
        self._pretty_print(tree.getroot())
        tree.write(filename)

    def _pretty_print(self, elem, level=0):
        indent = "  "  # 4 spaces
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = "\n" + indent * (level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = "\n" + indent * level
            for elem in elem:
                self._pretty_print(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = "\n" + indent * level
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = "\n" + indent * level

    def add_mesh(self, mesh: "GroupMesh"):
        """Add the mesh to the xdmf tree.

        Args:
            nodes_location: String to geometry/nodes in hdf file
            cells_location: String to topology/cells in hdf file
        """
        grid = ET.SubElement(
            self.domain, "Grid", Name=mesh._path.basename, GridType="Uniform"
        )
        with self._file.open(FileMode.READ):
            self._add_nodes(grid, mesh._path.joinpath("coordinates"))
            self._add_cells(grid, mesh._path.joinpath("topology"))

    def _add_nodes(self, grid: ET.Element, nodes_path: HDF5Path):
        geometry_type = "XY"

        points = cast(h5py.Dataset, self._file[nodes_path])
        geo = ET.SubElement(grid, "Geometry", GeometryType=geometry_type)
        dtype, precission = numpy_to_xdmf_dtype[points.dtype.name]
        dim = "{} {}".format(*points.shape)
        data_item = ET.SubElement(
            geo,
            "DataItem",
            DataType=dtype,
            Dimensions=dim,
            Format="HDF",
            Precision=precission,
        )
        data_item.text = f"{self._file._path.name}:{nodes_path}"

    def _add_cells(self, grid: ET.Element, cells_path: HDF5Path):
        cells = cast(h5py.Dataset, self._file[cells_path])
        nb_cells = cells.shape[0]
        topo = ET.SubElement(
            grid,
            "Topology",
            TopologyType=cells.attrs.get("cell_type", "Triangle"),
            NumberOfElements=str(nb_cells),
        )
        dim = "{} {}".format(*cells.shape)
        dt, prec = numpy_to_xdmf_dtype[cells.dtype.name]
        data_item = ET.SubElement(
            topo,
            "DataItem",
            DataType=dt,
            Dimensions=dim,
            Format="HDF",
            Precision=prec,
        )
        data_item.text = f"{self._file._path.name}:{cells_path}"

    def add_timeseries(
        self, timesteps: Iterable[float], fields: "list[FieldData]", mesh_name: str
    ):
        # if no timesteps, return
        if not timesteps:
            return

        collection = ET.SubElement(
            self.domain,
            "Grid",
            Name="TimeSeries",
            GridType="Collection",
            CollectionType="Temporal",
        )

        with self._file.open(FileMode.READ):
            for i, t in enumerate(timesteps):
                self._add_step(i, t, fields, collection, mesh_name)

    def _add_step(
        self,
        step: int,
        time: float,
        fields: "list[FieldData]",
        collection: ET.Element,
        mesh_name: str,
    ):
        """Write the data array for time t.

        Args:
            t (float): time
            data_location (str): String to data in h5 file
            name (str): Name for the field in the Xdmf file
        """
        # avoid time = NaN in xdmf
        if np.isnan(time):
            time = step

        grid = ET.SubElement(collection, "Grid")
        ptr = (
            f'xpointer(//Grid[@Name="{mesh_name}"]/*[self::Topology or self::Geometry])'
        )

        ET.SubElement(grid, "{http://www.w3.org/2003/XInclude}include", xpointer=ptr)
        ET.SubElement(grid, "Time", Value=str(time))

        for field in fields:
            self._add_attribute(grid, field, step)

    def _add_attribute(self, grid: ET.Element, field: "FieldData", step: int) -> None:
        """Write an attribute/field."""
        data = field._obj[str(step)]
        assert isinstance(data, h5py.Dataset), "Data is not a dataset"

        if data.ndim == 1 or data.shape[1] <= 1:
            att_type = "Scalar"
        elif data.ndim == 2:
            att_type = "Vector"
        elif data.ndim == 3 and len(set(data.shape[1:])) == 1:
            # Square shape -> Tensor
            att_type = "Tensor"
        else:
            att_type = "Matrix"

        # Cell or Node data
        data_type = FieldType(field._obj.attrs.get("field_type", FieldType.NODE))

        att = ET.SubElement(
            grid,
            "Attribute",
            Name=field.name,
            AttributeType=att_type,
            Center=data_type.value,
        )

        dt, prec = numpy_to_xdmf_dtype[data.dtype.name]
        dim = " ".join([str(i) for i in data.shape])

        data_item = ET.SubElement(
            att,
            "DataItem",
            DataType=dt,
            Dimensions=dim,
            Format="HDF",
            Precision=prec,
        )
        data_item.text = f"{self._file._path.name}:{field._path}/{step}"
