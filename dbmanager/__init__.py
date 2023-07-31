__author__ = 'florez@ethz.ch'
__copyright__ = ''
__license__ = 'LGPLv3'

from .manager import Manager
from .simulation import Simulation, SimulationReader, SimulationWriter
from .postprocessor import Postprocessor

import logging
import os
def log(mode: str = "INFO"):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", mode))
