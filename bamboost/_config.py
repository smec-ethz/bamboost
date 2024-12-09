"""The module includes functions to copy an example config file to the user's
config directory and load the config file from the directory. If the config
file does not exist, a copy of the example config file will be created.

Attributes:
    config: A config options object. Initialized from config file
        `~/.config/bamboost/config.toml`.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass, field, fields
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, get_type_hints

from bamboost import BAMBOOST_LOGGER as log

if TYPE_CHECKING:
    pass

if sys.version_info < (3, 11):
    import tomli  # type: ignore
else:
    import tomllib as tomli


__all__ = [
    "config",
]

CONFIG_DIR = Path("~/.config/bamboost").expanduser()
CONFIG_FILE = CONFIG_DIR.joinpath("config.toml")
LOCAL_DIR = Path("~/.local/share/bamboost").expanduser()
CACHE_DIR = Path("~/.cache/bamboost").expanduser()
DATABASE_FILE_NAME = "bamboost.db"


def _find_root_dir() -> Path:
    """Find the root directory."""

    ANCHORS = [
        ".git",
        "pyproject.toml",
    ]

    cwd = Path.cwd()
    for path in chain([cwd], cwd.parents):
        if any(path.joinpath(anchor).exists() for anchor in ANCHORS):
            return path

    log.warning("Root directory not found. Using current directory.")
    return cwd


ROOT_DIR = _find_root_dir()


def _get_global_config(filepath: Path) -> dict[str, Any]:
    """Reads the configuration file and fills the configuration options."""
    if not filepath.exists():
        log.warning("Config file not found. Using default settings.")
        return {}

    with filepath.open("rb") as f:
        try:
            return tomli.load(f)
        except tomli.TOMLDecodeError as e:
            log.error(f"Error reading config file: {e}")
            return {}


def _get_project_config() -> dict[str, Any]:
    """Get the project configuration."""
    with ROOT_DIR.joinpath("pyproject.toml").open("rb") as f:
        try:
            return tomli.load(f).get("tool", {}).get("bamboost", {})
        except tomli.TOMLDecodeError as e:
            log.error(f"Error reading project config file: {e}")
            return {}


@dataclass
class _Base:
    _field_aliases = {}

    def __repr__(self) -> str:
        s = ""
        max_length = max(len(field.name) for field in fields(self))
        for field in fields(self):
            s += f"{field.name:<{max_length+1}}: {getattr(self, field.name)}\n"
        return s

    def __getitem__(self, key: str) -> Any:
        """Access the configuration options by key, separated by dots."""
        try:
            current_selection = self
            for attr in key.split("."):
                current_selection = getattr(current_selection, attr)
            return current_selection
        except KeyError:
            raise KeyError(f"Invalid key path: '{key}'")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the configuration options by key, separated by dots."""
        current_selection = self
        keys = key.split(".")
        for attr in keys[:-1]:
            current_selection = getattr(current_selection, attr)
        setattr(current_selection, keys[-1], value)

    def __getattr__(self, name: str) -> Any:
        # handle aliases
        actual_name = self._field_aliases.get(name, name)
        return super().__getattribute__(actual_name)

    def _ipython_key_completions_(self) -> Generator[str, None, None]:
        for key, obj in self.__dict__.items():
            yield key
            for subkey in getattr(obj, "__dict__", {}).keys():
                yield f"{key}.{subkey}"

    @classmethod
    def from_dict(cls, config: dict):
        """Create an instance of the dataclass from a dictionary of
        configuration values.

        This method performs the following steps:
        1. Identifies valid field names and aliases.
        2. Logs any unknown configuration keys.
        3. Filters the input dictionary to include only valid keys and aliases.
        4. Validates the types of the configuration values.
        5. Creates an instance of the class with the filtered and validated configuration.
        6. Logs any missing fields that are using default values.

        Args:
            config: A dictionary containing configuration key-value pairs.

        Returns:
            An instance of the class initialized with the provided configuration.

        Raises:
            No exceptions are raised, but various warnings and errors are logged:
            - Unknown configuration keys are logged as info.
            - Invalid types for configuration values are logged as errors.
            - Type checking errors are logged as info.
            - Missing fields (using defaults) are logged as info.

        Note:
            - Invalid type values are removed from the configuration.
            - Aliases are resolved to their corresponding field names.
        """
        # Get the valid field names
        valid_fields = {f.name for f in fields(cls)}
        aliases = set(cls._field_aliases.keys())

        # Check for unknown keys
        unknown_keys = set(config) - valid_fields - aliases
        for key in unknown_keys:
            log.info(f"Unknown config key: {key}")

        # Filter the input dictionary to only include valid keys
        filtered_config = {k: v for k, v in config.items() if k in valid_fields}
        filtered_config.update(
            {cls._field_aliases[k]: v for k, v in config.items() if k in aliases}
        )

        # Validate type of user and project config values
        resolved_type_hints = get_type_hints(cls)
        for field_def in fields(cls):
            if field_def.name in filtered_config:
                try:
                    if not isinstance(
                        filtered_config[field_def.name],
                        resolved_type_hints[field_def.name],
                    ):
                        log.error(
                            (
                                f"Invalid type for config key '{field_def.name}': {filtered_config[field_def.name]}. "
                                f"Requires {field_def.type} "
                            )
                        )
                        # If the type is invalid, remove the key
                        filtered_config.pop(field_def.name)
                except TypeError:
                    # An error occurred while checking the type, keep the key but log a warning
                    log.info(
                        f"Error checking type for config key '{field_def.name}': {filtered_config[field_def.name]}"
                    )

        # Create the instance
        instance = cls(**filtered_config)

        # Check for missing fields (defaults used)
        for field_def in fields(cls):
            if field_def.name not in filtered_config:
                log.info(
                    f"Config key '{field_def.name}' not set; using default: {getattr(instance, field_def.name)}"
                )

        return instance


@dataclass(repr=False)
class _Paths(_Base):
    """Paths used by bamboost.

    This dataclass contains the paths used by bamboost.

    Attributes:
        localDir: The directory where the local data is stored.
        cacheDir: The directory where the cache is stored.
    """

    localDir: Path = field(default=LOCAL_DIR)
    cacheDir: Path = field(default=CACHE_DIR)

    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, str):
            value = Path(value).expanduser()
        super().__setattr__(name, value)


@dataclass(repr=False)
class _Options(_Base):
    """Core options for bamboost.

    This dataclass contains the core options for bamboost.

    Attributes:
        mpi: If True, the mpi4py package is available and used.
        sortTableKey: The default key to sort the table by.
        sortTableOrder: The default order to sort the table by.
    """

    mpi: bool = field(default=importlib.util.find_spec("mpi4py") is not None)
    sortTableKey: str = field(default="time_stamp")
    sortTableOrder: str = field(default="desc")


@dataclass(repr=False)
class _IndexOptions(_Base):
    """Index options for bamboost.

    This dataclass contains the index options for bamboost.

    Attributes:
        searchPaths: A list of paths to index.
        syncTables: If True, the sqlite tables are synchronized immediatly
            after some queries.
        convertArrays: If True, arrays are converted to np.arrays.
        databaseFileName: The name of the database file.
        databaseFile: The path to the database file.
        isolated: If true, this project manages it's own database. The
            searchPaths are reduced to the project root only.
    """

    _field_aliases = {
        "paths": "searchPaths",
    }

    searchPaths: list[Path] = field(default_factory=lambda: [Path("~").expanduser()])
    syncTables: bool = field(default=True)
    convertArrays: bool = True
    databaseFileName: str = field(default=DATABASE_FILE_NAME)
    databaseFile: Path = field(init=False)
    isolated: bool = False

    def __post_init__(self) -> None:
        # Parse search paths to Path objects
        self.searchPaths = [
            Path(p).expanduser() if isinstance(p, str) else p for p in self.searchPaths
        ]

        # Handle isolated mode
        if self.isolated:
            self.databaseFile = ROOT_DIR.joinpath(".bamboost.db")
            self.searchPaths = [ROOT_DIR]
        else:
            self.databaseFile = LOCAL_DIR.joinpath(self.databaseFileName)


@dataclass(repr=False, init=False)
class _Config(_Base):
    """Configuration class for bamboost.

    This class manages the configuration options and index settings for bamboost.
    It loads the configuration from a file and provides access to the options
    and index attributes.

    Attributes:
        paths: Paths used by bamboost.
        options: Configuration options for bamboost.
        index: Index settings for bamboost.
    """

    paths: _Paths
    options: _Options
    index: _IndexOptions

    def __init__(self) -> None:
        global_config = _get_global_config(CONFIG_FILE)
        project_config = _get_project_config()

        def nested_update(d: MutableMapping, u: MutableMapping) -> MutableMapping:
            for k, v in u.items():
                d[k] = (
                    nested_update(d.get(k, {}), v)
                    if isinstance(v, MutableMapping)
                    else v
                )
            return d

        config = nested_update(global_config, project_config)

        self.paths = _Paths.from_dict(config.pop("paths", {}))
        self.options = _Options.from_dict(config.pop("options", {}))
        self.index = _IndexOptions.from_dict(config.pop("index", {}))

        # Log unknown config options
        for key in global_config.keys():
            log.info(f"Unknown config table: {key}")

    def __repr__(self) -> str:
        s = str()
        for field in fields(self):
            s += f"> {field.name.upper()}\n"
            s += getattr(self, field.name).__repr__()
        return s


# Create the config instance
config: _Config = _Config()
