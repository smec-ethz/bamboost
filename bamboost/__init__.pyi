import logging
from typing import Literal

__author__: str
__copyright__: str
__license__: str
__version__: str

BAMBOOST_LOGGER: logging.Logger

def set_log_level(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None: ...

from bamboost._config import config as config

from bamboost.core.manager import *
from bamboost.core.simulation import *
