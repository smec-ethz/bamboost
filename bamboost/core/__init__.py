from typing import TYPE_CHECKING

import lazy_loader as _lazy

if TYPE_CHECKING:
    from .collection import Collection as Collection
    from .simulation import FieldType as FieldType
    from .simulation import Simulation as Simulation
    from .simulation import SimulationWriter as SimulationWriter

__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    [],
    {
        "collection": ["Collection"],
        "simulation": ["Simulation", "SimulationWriter", "FieldType"],
    },
)
