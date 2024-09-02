__author__ = "florez@ethz.ch"
__copyright__ = ""
__license__ = "LGPLv3"

import logging
from typing import Literal

# Determine the version of the package
try:
    # If the package is installed, the version is stored in _version.py
    from bamboost._version import __version__
except ImportError:
    # This is necessary if the package is not installed
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)


def add_stream_handler(logger: logging.Logger) -> None:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: %(levelname)s - %(message)s",
        style="%",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


BAMBOOST_LOGGER = logging.getLogger("bamboost")
add_stream_handler(BAMBOOST_LOGGER)


def set_log_level(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None:
    BAMBOOST_LOGGER.setLevel(level)


from bamboost._config import config  # noqa: E402, F401
from bamboost.extensions import extensions  # noqa: E402, F401
from bamboost.manager import Manager  # noqa: E402, F401
from bamboost.simulation import Simulation  # noqa: E402, F401
from bamboost.simulation_writer import SimulationWriter  # noqa: E402, F401
