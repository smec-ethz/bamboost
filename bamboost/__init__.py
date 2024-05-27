__author__ = "florez@ethz.ch"
__copyright__ = ""
__license__ = "LGPLv3"
__version__ = "0.6.0"

import logging
import os

from ._config import config
from .extensions import extensions


def _lazy_load_stub_manager(*args, **kwargs):
    global Manager
    from .manager import Manager

    return Manager(*args, **kwargs)


Manager = _lazy_load_stub_manager


def _lazy_load_stub_simulation(*args, **kwargs):
    global Simulation
    from .simulation import Simulation

    return Simulation(*args, **kwargs)


Simulation = _lazy_load_stub_simulation


def _lazy_load_stub_simulation_writer(*args, **kwargs):
    global SimulationWriter
    from .simulation_writer import SimulationWriter

    return SimulationWriter(*args, **kwargs)


SimulationWriter = _lazy_load_stub_simulation_writer


def set_log_level(level: int = 30):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", level))


# Set default logging level to WARNING (30)
set_log_level(30)
