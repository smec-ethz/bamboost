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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    pass

if sys.version_info < (3, 11):
    import tomli  # type: ignore
else:
    import tomllib as tomli

from bamboost import BAMBOOST_LOGGER as log

log.setLevel("DEBUG")

__all__ = [
    "config",
]


def _config_repr(self) -> str:
    s = ""
    max_length = max(len(attr) for attr in self.__annotations__.keys())
    for attr in self.__annotations__.keys():
        value = getattr(self, attr)
        s += f"{attr:<{max_length+1}}: {value}\n"
    return s


@dataclass
class _Paths:
    """Paths used by bamboost.

    This dataclass contains the paths used by bamboost.

    Attributes:
        home: The user's home directory.
        configDir: The directory where the config file is stored.
        configFile: The path to the config file.
        localDir: The directory where the local data is stored.
        databaseFile: The path to the database file.
        cacheDir: The directory where the cache is stored.
    """

    home: Path = field(
        default=Path("~").expanduser(),
    )
    configDir: Path = field(
        default=Path("~/.config/bamboost").expanduser(),
    )
    configFile: Path = field(
        default=Path("~/.config/bamboost/config.toml").expanduser(),
    )
    localDir: Path = field(
        default=Path("~/.local/share/bamboost").expanduser(),
    )
    databaseFile: Path = field(
        default=Path("~/.local/share/bamboost/bamboost.db").expanduser(),
    )
    cacheDir: Path = field(
        default=Path("~/.cache/bamboost").expanduser(),
    )

    __repr__ = _config_repr

    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, str):
            value = Path(value).expanduser()
        super().__setattr__(name, value)


@dataclass
class _DatabaseOptions:
    """Core options for bamboost.

    This dataclass contains the core options for bamboost.

    Attributes:
        sortTableKey: The default key to sort the table by.
        sortTableOrder: The default order to sort the table by.
    """

    sortTableKey: str = field(
        default="time_stamp",
    )
    sortTableOrder: str = field(
        default="desc",
    )

    __repr__ = _config_repr


@dataclass
class _IndexOptions:
    """Index options for bamboost.

    This dataclass contains the index options for bamboost.

    Attributes:
        paths: A list of paths to index.
        databaseFile: The path to the database file.
        syncTables: If True, the sqlite tables are synchronized immediatly
            after some queries.
    """

    paths: list[Path] = field(
        default_factory=lambda: [Path("~").expanduser()],
    )
    databaseFile: Path = field(
        default=Path("~/.local/share/bamboost/bamboost.db").expanduser(),
    )
    syncTables: bool = field(
        default=True,
    )

    __repr__ = _config_repr


class _Config:
    """Configuration class for bamboost.

    This class manages the configuration options and index settings for bamboost.
    It loads the configuration from a file and provides access to the options
    and index attributes.

    Attributes:
        options: Configuration options for bamboost.
        index: Index settings for bamboost.
    """

    mpi: bool
    paths: _Paths
    table: _DatabaseOptions
    index: _IndexOptions

    def __init__(self) -> None:
        user_config = self.read_config_file(_Paths.configFile)

        self.mpi = user_config.pop(
            "mpi", importlib.util.find_spec("mpi4py") is not None
        )
        self.paths = _Paths(**user_config.pop("paths", {}))
        self.table = _DatabaseOptions(**user_config.pop("table", {}))
        self.index = _IndexOptions(**user_config.pop("index", {}))

        # Log unknown config options
        for key, value in user_config.items():
            log.info(f"Unknown config option: {key}={value}")

    def __getitem__(self, key: str) -> Any:
        """Access the configuration options by key, separated by dots."""
        current_selection = self
        for attr in key.split("."):
            current_selection = getattr(current_selection, attr)
        return current_selection

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the configuration options by key, separated by dots."""
        current_selection = self
        keys = key.split(".")
        for attr in keys[:-1]:
            current_selection = getattr(current_selection, attr)
        setattr(current_selection, keys[-1], value)

    def _ipython_key_completions_(self) -> Generator[str, None, None]:
        for key, obj in self.__dict__.items():
            yield key
            for subkey in getattr(obj, "__dict__", {}).keys():
                yield f"{key}.{subkey}"

    def __repr__(self) -> str:
        s = "Bamboost Configuration\n----------------------"
        for key, obj in self.__dict__.items():
            s += f"\n{key}:\t"
            obj_repr = obj.__repr__()
            s += obj_repr.replace("\n", "\n\t")
        return s

    def read_config_file(self, filepath: Path) -> dict[str, Any]:
        """Reads the configuration file and fills the configuration options."""
        if not filepath.exists():
            log.warning("Config file not found. Using default settings.")
            return {}

        with filepath.open("rb") as f:
            return tomli.load(f)


# Create the config instance
config: _Config = _Config()
