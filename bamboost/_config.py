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
    - config: A dictionary of the config. Initiated from config file.
"""

import os
import pkgutil

try:
    import tomllib as tomli
except ImportError:
    import tomli

__all__ = ["paths", "config"]

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

# Create directories if they do not exist
os.makedirs(paths["CONFIG_DIR"], exist_ok=True)
os.makedirs(paths["LOCAL_DIR"], exist_ok=True)


def _copy_config_file(file: str):
    """Copy the example config file to the user's config directory.

    Raises:
        - FileExistsError: If the config file already exists in the destination
          directory."""
    # If file exists, raise an error
    if os.path.exists(file):
        raise FileExistsError("Config file already exists")

    # source_file = "_example_config.toml"
    source_file: bytes = pkgutil.get_data("bamboost", "_example_config.toml")

    # Copy the source file to the destination
    with open(file, "wb") as f:
        f.write(source_file)


def _load_config_file(file: str = None):
    """Load the config file from the user's config directory.

    If the config file does not exist, a copy of the example config file will
    be created.

    Returns:
        - dict: The contents of the config file as a dictionary.
    """
    # Load the config file
    if file is None:
        file = paths["CONFIG_FILE"]
    try:
        with open(file, "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        _copy_config_file(file)
        return _load_config_file(file)


config = _load_config_file()
