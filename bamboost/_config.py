"""
The module includes functions to copy an example config file to the user's
config directory and load the config file from the directory. If the config
file does not exist, a copy of the example config file will be created.

Functions:
    - _copy_config_file(): Copies the example config file to the user's config
      directory.
    - _load_config_file(): Loads the config file from the user's config
      directory.

Constants:
    - HOME: The user's home directory.
    - CONFIG_DIR: The path to the bamboost config directory.
    - LOCAL_DIR: The path to the local bamboost directory.
    - CONFIG_FILE: The path to the config file.
    - DATABASE_FILE: The path to the database file.
"""

import os
import pkgutil
import shutil

try:
    import tomllib as tomli
except ImportError:
    import tomli

# Define paths to bamboost config files
HOME = os.path.expanduser("~")
CONFIG_DIR = os.path.join(HOME, ".config", "bamboost")
LOCAL_DIR = os.path.join(HOME, ".local", "share", "bamboost")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.toml")
DATABASE_FILE = os.path.join(LOCAL_DIR, "bamboost.db")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(LOCAL_DIR, exist_ok=True)


def _copy_config_file():
    """Copy the example config file to the user's config directory.

    Raises:
        - FileExistsError: If the config file already exists in the destination
          directory. """
    # If file exists, raise an error
    if os.path.exists(CONFIG_FILE):
        raise FileExistsError("Config file already exists")

    # source_file = "_example_config.toml"
    source_file: bytes = pkgutil.get_data("bamboost", "_example_config.toml")

    # Copy the source file to the destination
    with open(CONFIG_FILE, "wb") as f:
        f.write(source_file)


def _load_config_file():
    """Load the config file from the user's config directory.

    If the config file does not exist, a copy of the example config file will
    be created.

    Returns:
        - dict: The contents of the config file as a dictionary.
    """
    # Load the config file
    try:
        with open(CONFIG_FILE, "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        _copy_config_file()
        return _load_config_file()


config = _load_config_file()
options = config.get("options", {})
options["sync_tables"] = True
