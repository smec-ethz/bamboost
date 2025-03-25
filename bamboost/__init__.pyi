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
from bamboost.core.collection import *
from bamboost.core.simulation import *
from bamboost.index import Index as Index
