from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import lazy_loader as _lazy

from bamboost import plugins as plugins
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
    from bamboost.core.collection import Collection as Collection
    from bamboost.core.simulation import FieldType as FieldType
    from bamboost.core.simulation import Simulation as Simulation
    from bamboost.core.simulation import SimulationWriter as SimulationWriter
    from bamboost.index import Index as Index

# We use lazy_loader to avoid upfront imports of submodules while still
# providing a consistent API for the user.
__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    [],
    {
        "core.collection": ["Collection"],
        "core.simulation": ["Simulation", "SimulationWriter", "FieldType"],
        "index": ["Index"],
    },
)

# by default, we set the log level to INFO and add a stream handler to the BAMBOOST_LOGGER
# this ensures that log messages are printed to the console by default
add_stream_handler(BAMBOOST_LOGGER)
BAMBOOST_LOGGER.setLevel(config.options.logLevel)
