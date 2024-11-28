"""
The module includes functions to copy an example config file to the user's
config directory and load the config file from the directory. If the config
file does not exist, a copy of the example config file will be created.

Functions:
    - _copy_config_file(): Copies the example config file to the user's config
      directory.
    - _load_config_file(): Loads the config file from the user's config
      directory.

Attributes:
    - paths: A dictionary containing paths to the config files.
    - config: A config options object. Initialized from config file `~/.config/bamboost/config.toml`.
"""

import importlib.util
import os
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Union

try:
    import tomllib as tomli
except ImportError:
    import tomli

__all__ = [
    "paths",
    "config",
    "Config",
]

# Define paths to bamboost config files
_home_dir = os.path.expanduser("~")
paths = {
    "HOME": os.path.expanduser("~"),
    "CONFIG_DIR": os.path.join(_home_dir, ".config", "bamboost"),
    "LOCAL_DIR": os.path.join(_home_dir, ".local", "share", "bamboost"),
    "CONFIG_FILE": os.path.join(_home_dir, ".config", "bamboost", "config.toml"),
    "DATABASE_FILE": os.path.join(
        _home_dir, ".local", "share", "bamboost", "bamboost.db"
    ),
}


@dataclass
class ConfigOptions:
    """Default options for bamboost."""

    mpi: bool = True
    sort_table_key: str = "time_stamp"
    sort_table_order: str = "desc"
    sync_tables: bool = True


@dataclass
class ConfigIndex:
    """Default index options for bamboost."""

    paths: list = field(default_factory=list)


class Config:
    """Configuration class for bamboost.

    This class manages the configuration options and index settings for bamboost.
    It loads the configuration from a file and provides access to the options
    and index attributes.

    Attributes:
        options: Configuration options for bamboost.
        index: Index settings for bamboost.
    """

    options: ConfigOptions
    index: ConfigIndex

    def __init__(self, config_file: str = None) -> None:
        self._config_file = config_file or paths["CONFIG_FILE"]
        loaded_config = self._load_config_file(self._config_file)

        self.options = ConfigOptions(**loaded_config.get("options", {}))
        self.index = ConfigIndex(**loaded_config.get("index", {}))

        # Disable MPI if not available
        self.options.mpi = (
            self.options.mpi and importlib.util.find_spec("mpi4py") is not None
        )

    def _load_config_file(self, file: str = None) -> Union[dict, None]:
        """If the file exists, loads the config file from the user's config
        directory.

        Returns:
            Config object: The contents of the config file as a dictionary. Or None.
        """
        # Load the config file
        if file is None:
            file = paths["CONFIG_FILE"]

        if os.path.exists(file):
            with open(file, "rb") as f:
                return tomli.load(f)
        else:
            return {}

    def __repr__(self) -> str:
        return dedent(f"""\
            Config(
                options={self.options.__dict__}, 
                index={self.index.__dict__}
            )""")


config: Config = Config()
