import logging
from typing import Literal

__author__: str
__copyright__: str
__license__: str
__version__: str

BAMBOOST_LOGGER: logging.Logger
STREAM_HANDLER: logging.StreamHandler

def set_log_level(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None: ...

from bamboost import plugins as plugins
from bamboost._config import config as config
from bamboost.core.collection import Collection as Collection
from bamboost.core.simulation import FieldType as FieldType
from bamboost.core.simulation import Simulation as Simulation
from bamboost.core.simulation import SimulationWriter as SimulationWriter
from bamboost.index import Index as Index
