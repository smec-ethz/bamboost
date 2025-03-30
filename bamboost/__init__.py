import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import lazy_loader as lazy

__author__ = "florez@ethz.ch"
__copyright__ = ""
__license__ = "MIT"
try:
    __version__ = version("bamboost")
except PackageNotFoundError:  # not installed
    __version__ = "unknown"

BAMBOOST_LOGGER = logging.getLogger("bamboost")
STREAM_HANDLER = logging.StreamHandler()

from bamboost import plugins as plugins
from bamboost._config import config as config


def _add_stream_handler(logger: logging.Logger) -> None:
    from bamboost.mpi import MPI, MPI_ON

    class _LogFormatterWithRank(logging.Formatter):
        def format(self, record):
            record.rank = MPI.COMM_WORLD.rank
            return super().format(record)

    if MPI_ON:
        formatter = _LogFormatterWithRank(
            "[%(asctime)s] %(name)s: %(levelname)s [%(rank)d] - %(message)s",
            style="%",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s: %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    STREAM_HANDLER.setFormatter(formatter)
    logger.addHandler(STREAM_HANDLER)


_add_stream_handler(BAMBOOST_LOGGER)


def set_log_level(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None:
    BAMBOOST_LOGGER.setLevel(level)


# We use lazy_loader to avoid upfront imports of submodules while still
# providing a consistent API for the user.
# BUT: keep stub file of this module up-to-date with the actual imports
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    [],
    {
        "core.collection": ["Collection"],
        "core.simulation": ["Simulation", "SimulationWriter", "FieldType"],
        "index": ["Index"],
    },
)
