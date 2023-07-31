__author__ = 'florez@ethz.ch'
__copyright__ = ''
__license__ = 'LGPLv3'

from .manager import Manager
from .simulation import Simulation, SimulationWriter
from .reader import SimulationReader
from .postprocessor import Postprocessor


import logging
import os

def set_log_level(level: int = 30):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", level))

# Set default logging level to WARNING (30)
set_log_level(30)
