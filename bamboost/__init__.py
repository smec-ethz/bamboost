__author__ = "florez@ethz.ch"
__copyright__ = ""
__license__ = "LGPLv3"
__version__ = "0.4.5"

import logging
import os

from .manager import Manager
from .simulation import Simulation
from .simulation_writer import SimulationWriter


def set_log_level(level: int = 30):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", level))


# Set default logging level to WARNING (30)
set_log_level(30)
