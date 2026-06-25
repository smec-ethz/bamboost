"""
This module manages configuration options for bamboost. It supports loading configuration
from a global file (`~/.config/bamboost/config.toml`) and a project configuration in the
standard (`pyproject.toml`).

The configuration is structured using dataclasses, allowing configuration handling.

Key Features:
- Detects the root directory of the project based on common anchor files.
- Reads configuration from global and project-specific TOML files.

Attributes:
    ROOT_DIR (Path): The detected root directory of the project.
    config (_Config): The main configuration instance containing paths, options, and index
        settings.

"""

import sys
from dataclasses import dataclass, field, fields
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
)

from bamboost._logger import BAMBOOST_LOGGER as log
from bamboost._typing import StrPath
from bamboost.constants import DEFAULT_CONFIG_FILE_NAME

if sys.version_info >= (3, 11):
    import tomllib as tomli
else:
    import tomli  # type: ignore


__all__ = [
    "config",
]

CONFIG_DIR = Path("~/.config/bamboost").expanduser()
CONFIG_FILE = CONFIG_DIR.joinpath(DEFAULT_CONFIG_FILE_NAME)


def _find_root_dir() -> Optional[Path]:
    """Find the root directory."""

    ANCHORS = [
        ".git",
        "pyproject.toml",
    ]

    cwd = Path.cwd()
    try:
        return next(
            path
            for path in chain([cwd], cwd.parents)
            if any(path.joinpath(anchor).exists() for anchor in ANCHORS)
        )
    except StopIteration:
        log.info("Root directory not found.")
        return None


def _get_global_config(filepath: Path) -> dict[str, Any]:
    """Reads the configuration file and fills the configuration options."""
    try:
        with filepath.open("rb") as f:
            try:
                return tomli.load(f)
            except tomli.TOMLDecodeError as e:
                log.warning(f"Error reading config file: {e}")
                return {}
    except FileNotFoundError:
        log.info("Config file not found or unreadable. Using default settings.")
        return {}


def _get_project_config(project_dir: Path) -> dict[str, Any]:
    """Get the project configuration from bamboost.toml or pyproject.toml."""
    bamboost_path = project_dir.joinpath("bamboost.toml")
    pyproject_path = project_dir.joinpath("pyproject.toml")

    if bamboost_path.is_file():
        try:
            with bamboost_path.open("rb") as f:
                return tomli.load(f)
        except tomli.TOMLDecodeError as e:
            log.warning(f"Error reading bamboost.toml: {e}")
            return {}

    if pyproject_path.is_file():
        try:
            with pyproject_path.open("rb") as f:
                return tomli.load(f).get("tool", {}).get("bamboost", {})
        except tomli.TOMLDecodeError as e:
            log.warning(f"Error reading pyproject.toml: {e}")
            return {}

    log.info("No configuration file found. Using default settings.")
    return {}


@dataclass(repr=False)
class _Config:
    """Configuration class for bamboost.

    This class manages the configuration options and index settings for bamboost.
    It loads the configuration from a file and provides access to the options
    and index attributes.

    Args:
        project_dir: An optional alternative directory to load the project-based config
            from.
    """

    mpi: bool = field(default=False)
    sort_table_key: str = field(default="created_at")
    sort_table_order: str = field(default="desc")
    log_file_lock_severity: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        field(default="WARNING")
    )
    file_lock_timeout: float | None = field(default=60.0)
    log_root_only: bool = field(default=False)
    logLevel: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = field(
        default="WARNING"
    )

    def __setattr__(self, name: str, value: Any, /) -> None:
        super().__setattr__(name, value)
        if name == "mpi" and "bamboost.mpi" in sys.modules:
            try:
                from bamboost.mpi import _MPIProxy

                _MPIProxy.set_from_ctx()
            except ImportError:
                pass

    def __init__(self, project_dir: Optional[StrPath] = None) -> None:
        global_config = _get_global_config(CONFIG_FILE)
        project_dir = Path(project_dir).expanduser() if project_dir else None
        project_dir = project_dir or _find_root_dir()
        if project_dir:
            project_config = _get_project_config(project_dir)
        else:
            project_config = {}

        config_dict = global_config.copy()
        config_dict.update(project_config)

        # Flatten options
        options_dict = config_dict.pop("options", {})
        config_dict.update(options_dict)

        valid_fields = {f.name for f in fields(self)}

        for key, value in config_dict.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                log.info(f"Unknown config table: {key}")

    def __repr__(self) -> str:
        s = str()
        for f in fields(self):
            s += f"{f.name}: {getattr(self, f.name)}\n"
        return s


# Create the config instance (load the configuration)
config: _Config = _Config()
