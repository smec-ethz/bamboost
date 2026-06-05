from typing import TYPE_CHECKING

import lazy_loader as _lazy

from bamboost import plugins as plugins
from bamboost._config import config as config
from bamboost._logger import BAMBOOST_LOGGER, add_stream_handler

__author__: str = "florez@ethz.ch"
__copyright__: str = ""
__license__: str = "MIT"


if TYPE_CHECKING:
    from bamboost.core.collection import Collection as Collection
    from bamboost.core.simulation import FieldType as FieldType
    from bamboost.core.simulation import Simulation as Simulation
    from bamboost.core.simulation import SimulationWriter as SimulationWriter
    from bamboost.index import Index as Index

# We use lazy_loader to avoid upfront imports of submodules while still
# providing a consistent API for the user.
_lazy_getattr, _lazy_dir, __all__ = _lazy.attach(
    __name__,
    [],
    {
        "core.collection": ["Collection"],
        "core.simulation": ["Simulation", "SimulationWriter", "FieldType"],
        "index": ["Index"],
    },
)


def __getattr__(name: str):
    if name == "__version__":
        try:
            from importlib.metadata import version

            return version("bamboost")
        except Exception:
            return "unknown"
    return _lazy_getattr(name)


def __dir__():
    return _lazy_dir() + ["__version__"]


# by default, we set the log level to INFO and add a stream handler to the BAMBOOST_LOGGER
# this ensures that log messages are printed to the console by default
add_stream_handler(BAMBOOST_LOGGER)
BAMBOOST_LOGGER.setLevel(config.options.logLevel)
