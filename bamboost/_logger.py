import logging
from typing import Literal

BAMBOOST_LOGGER: logging.Logger = logging.getLogger("bamboost")
STREAM_HANDLER: logging.StreamHandler = logging.StreamHandler()


def add_stream_handler(logger: logging.Logger) -> None:
    from bamboost.mpi import COMM_WORLD, MPI_ON

    class _LogFormatterWithRank(logging.Formatter):
        def format(self, record):
            record.rank = COMM_WORLD.rank
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


def set_log_level(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None:
    BAMBOOST_LOGGER.setLevel(level)
