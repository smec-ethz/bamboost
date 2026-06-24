from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import lazy_loader as _lazy

from bamboost._config import config as config
from bamboost._logger import BAMBOOST_LOGGER, add_stream_handler

__author__: str = "florez@ethz.ch"
__copyright__: str = ""
__license__: str = "MIT"
__version__: str
try:
    __version__ = version("bamboost")
except PackageNotFoundError:  # not installed
    __version__ = "unknown"


if TYPE_CHECKING:
    from bamboost.collection import Collection as Collection
    from bamboost.simulation import FieldType as FieldType
    from bamboost.simulation import Simulation as Simulation
    from bamboost.simulation import SimulationWriter as SimulationWriter

# We use lazy_loader to avoid upfront imports of submodules while still
# providing a consistent API for the user.
__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    [],
    {
        "collection": ["Collection"],
        "simulation": ["Simulation", "SimulationWriter", "FieldType"],
    },
)

# by default, we set the log level to INFO and add a stream handler to the BAMBOOST_LOGGER
# this ensures that log messages are printed to the console by default
add_stream_handler(BAMBOOST_LOGGER)
BAMBOOST_LOGGER.setLevel(config.options.logLevel)
