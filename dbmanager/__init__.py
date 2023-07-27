__author__ = 'florez@ethz.ch'
__copyright__ = ''
__license__ = 'LGPLv3'

from .manager import Manager
from .simulation import Simulation, SimulationReader, SimulationWriter
from .postprocessor import Postprocessor

import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
