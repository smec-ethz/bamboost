# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

import os
import xml.etree.ElementTree as ET

from bamboost.common.file_handler import FileHandler

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


class XDMFWriter:
    """Write xdmf file for a subset of the stored data in the H5 file.

    Args:
        filename (str): xdmf file path
        h5file (str): h5 file path"""

    def __init__(self, filename: str, _file: FileHandler):
        self.filename = filename
        self._file = _file
        self.h5file = os.path.basename(_file.file_name)
        self.xdmf_file = ET.Element("Xdmf", Version="3.0")
        self.domain = ET.SubElement(self.xdmf_file, "Domain")
        ET.register_namespace("xi", "https://www.w3.org/2001/XInclude/")
        self.mesh_name = "mesh"

    def write_file(self):
        tree = ET.ElementTree(self.xdmf_file)
        self._pretty_print(tree.getroot())
        tree.write(self.filename)

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

    def write_points_cells(self, points_location: str, cells_location: str):
        """Write the mesh to the xdmf file.

        Args:
            points (str): String to geometry/nodes in h5 file
            cells (str): String to topology/cells in h5 file
        """
        grid = ET.SubElement(
            self.domain, "Grid", Name=self.mesh_name, GridType="Uniform"
        )
        self._points(grid, points_location)
        self._cells(grid, cells_location)

    def _points(self, grid: ET.Element, points_location: str):
        geometry_type = "XY"

        with self._file("r") as _f:
            f = _f.file_object
            points = f[points_location]
            geo = ET.SubElement(grid, "Geometry", GeometryType=geometry_type)
            dt, prec = numpy_to_xdmf_dtype[points.dtype.name]
            dim = "{} {}".format(*points.shape)
            data_item = ET.SubElement(
                geo,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format="HDF",
                Precision=prec,
            )
            data_item.text = f"{self.h5file}:/{points_location}"

    def _cells(self, grid: ET.Element, cells_location: str):
        with self._file("r") as _f:
            f = _f.file_object
            cells = f[cells_location]
            nb_cells = cells.shape[0]
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType="Triangle",
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
            data_item.text = f"{self.h5file}:/{cells_location}"

    def add_timeseries(self, steps: int, fields: list):
        collection = ET.SubElement(
            self.domain,
            "Grid",
            Name="TimeSeries",
            GridType="Collection",
            CollectionType="Temporal",
        )

        for i in range(steps):
            self.write_step(collection, fields, i)

    def write_step(self, collection: ET.Element, fields: list, step: int):
        """Write the data array for time t.

        Args:
            t (float): time
            data_location (str): String to data in h5 file
            name (str): Name for the field in the Xdmf file
        """
        with self._file("r") as _f:
            f = _f.file_object
            grid = ET.SubElement(collection, "Grid")
            ptr = f'xpointer(//Grid[@Name="{self.mesh_name}"]/*[self::Topology or self::Geometry])'

            ET.SubElement(
                grid, "{http://www.w3.org/2003/XInclude}include", xpointer=ptr
            )

            t = f[f"data/{fields[0]}/{step}"].attrs.get("t", step)
            ET.SubElement(grid, "Time", Value=str(t))

            for name in fields:
                self.write_attribute(grid, name, name, step)

    def write_attribute(
        self, grid: ET.Element, field_name: str, name: str, step: int
    ) -> None:
        """Write an attribute/field."""
        with self._file("r") as _f:
            f = _f.file_object
            data = f[f"data/{field_name}/{step}"]

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
            center = data.attrs.get("center", "Node")

            att = ET.SubElement(
                grid,
                "Attribute",
                Name=name,
                AttributeType=att_type,
                Center=center,
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
            data_item.text = f"{self.h5file}:/data/{field_name}/{step}"
