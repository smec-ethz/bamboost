import logging
from importlib.metadata import version
from typing import Literal

__author__ = "florez@ethz.ch"
__copyright__ = ""
__license__ = "MIT"
__version__ = version("bamboost")


def _add_stream_handler(logger: logging.Logger) -> None:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: %(levelname)s - %(message)s",
        style="%",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


BAMBOOST_LOGGER = logging.getLogger("bamboost")
_add_stream_handler(BAMBOOST_LOGGER)


def set_log_level(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None:
    BAMBOOST_LOGGER.setLevel(level)


from bamboost._config import config
