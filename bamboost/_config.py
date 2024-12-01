"""The module includes functions to copy an example config file to the user's
config directory and load the config file from the directory. If the config
file does not exist, a copy of the example config file will be created.

Attributes:
    config: A config options object. Initialized from config file
        `~/.config/bamboost/config.toml`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterable

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

    @classmethod
    def from_dict(cls, config: dict):
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

    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, str):
            value = Path(value).expanduser()
        super().__setattr__(name, value)


@dataclass(repr=False)
class _Options(_Base):
    """Core options for bamboost.

    This dataclass contains the core options for bamboost.

    Attributes:
        sortTableKey: The default key to sort the table by.
        sortTableOrder: The default order to sort the table by.
    """

    mpi: bool = field(
        default=importlib.util.find_spec("mpi4py") is not None,
    )
    sortTableKey: str = field(
        default="time_stamp",
    )
    sortTableOrder: str = field(
        default="desc",
    )


@dataclass(repr=False)
class _IndexOptions(_Base):
    """Index options for bamboost.

    This dataclass contains the index options for bamboost.

    Attributes:
        paths: A list of paths to index.
        databaseFile: The path to the database file.
        syncTables: If True, the sqlite tables are synchronized immediatly
            after some queries.
    """

    _field_aliases = {
        "paths": "searchPaths",
    }

    searchPaths: list[Path] = field(
        default_factory=lambda: [Path("~").expanduser()],
    )
    databaseFile: Path = field(
        default=Path("~/.local/share/bamboost/bamboost.db").expanduser(),
    )
    syncTables: bool = field(
        default=True,
    )


@dataclass(repr=False, init=False)
class _Config(_Base):
    """Configuration class for bamboost.

    This class manages the configuration options and index settings for bamboost.
    It loads the configuration from a file and provides access to the options
    and index attributes.

    Attributes:
        options: Configuration options for bamboost.
        index: Index settings for bamboost.
    """

    paths: _Paths
    options: _Options
    index: _IndexOptions

    def __init__(self) -> None:
        user_config = self.read_config_file(_Paths.configFile)

        self.paths = _Paths.from_dict(user_config.pop("paths", {}))
        self.options = _Options.from_dict(user_config.pop("options", {}))
        self.index = _IndexOptions.from_dict(user_config.pop("index", {}))

        # Log unknown config options
        for key in user_config.keys():
            log.info(f"Unknown config table: {key}")

    def __repr__(self) -> str:
        s = str()
        for field in fields(self):
            s += f"> {field.name.upper()}\n"
            s += getattr(self, field.name).__repr__()
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
