__author__ = "florez@ethz.ch"
__copyright__ = ""
__license__ = "LGPLv3"
__version__ = "0.5.2"

import logging
import os

try:
    import tomllib as toml
except ImportError:
    import tomli as toml

HOME = os.path.expanduser("~")
CONFIG_DIR = os.path.join(HOME, ".config", "bamboost")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.toml")
try:
    with open(CONFIG_FILE, "rb") as f:
        config = toml.load(f)
except FileNotFoundError:
    config = {}


from .manager import Manager
from .simulation import Simulation
from .simulation_writer import SimulationWriter
from .index import get_path, get_known_paths, get_index_dict, get_uid_from_path


def set_log_level(level: int = 30):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", level))


# Set default logging level to WARNING (30)
set_log_level(30)
