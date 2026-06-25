from importlib.metadata import PackageNotFoundError, version

from bamboost._config import config as config
from bamboost._logger import BAMBOOST_LOGGER, add_stream_handler
from bamboost.collection import Collection
from bamboost.simulation import FieldType, Simulation, SimulationWriter

__author__: str = "florez@ethz.ch"
__copyright__: str = ""
__license__: str = "MIT"
__version__: str

try:
    __version__ = version("bamboost")
except PackageNotFoundError:  # not installed
    __version__ = "unknown"


# by default, we set the log level to INFO and add a stream handler to the BAMBOOST_LOGGER
# this ensures that log messages are printed to the console by default
add_stream_handler(BAMBOOST_LOGGER)
BAMBOOST_LOGGER.setLevel(config.options.logLevel)

__all__ = [
    "BAMBOOST_LOGGER",
    "Collection",
    "FieldType",
    "Simulation",
    "SimulationWriter",
    "config",
]
