from enum import Enum


class FieldType(Enum):
    NODE = "Node"
    ELEMENT = "Cell"


class CellType(Enum):
    # 2D types
    VERTEX = "Vertex"
    LINE = "Line"
    TRIANGLE = "Triangle"
    QUAD = "Quadrilateral"

    # 3D types
    TETRAHEDRON = "Tetrahedron"
    HEXAHEDRON = "Hexahedron"
    WEDGE = "Wedge"


from .base import Simulation as Simulation
from .base import SimulationWriter as SimulationWriter
