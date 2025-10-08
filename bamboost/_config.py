"""
This module manages configuration options for bamboost. It supports loading configuration
from a global file (`~/.config/bamboost/config.toml`) and a project configuration in the
standard (`pyproject.toml`).

The configuration is structured using dataclasses, allowing hierarchical and
type-validated configuration handling.

Key Features:
- Detects the root directory of the project based on common anchor files.
- Reads configuration from global and project-specific TOML files.
- Provides structured access to configuration values via dataclasses.
- Supports nested dictionary updates for merging configuration sources.
- Includes an index system for managing paths and database settings.

Attributes:
    ROOT_DIR (Path): The detected root directory of the project.
    config (_Config): The main configuration instance containing paths, options, and index
        settings.

"""

import importlib.util
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass, field, fields
from itertools import chain
from os import makedirs
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
    Literal,
    Optional,
    Union,
    get_type_hints,
)

from bamboost import BAMBOOST_LOGGER as log
from bamboost._typing import StrPath
from bamboost.utilities import PathSet

if TYPE_CHECKING:
    from typing_extensions import Self

if sys.version_info < (3, 11):
    import tomli  # type: ignore
else:
    import tomllib as tomli


__all__ = [
    "config",
]

CONFIG_DIR = Path("~/.config/bamboost").expanduser()
CONFIG_FILE = CONFIG_DIR.joinpath("config-next.toml")
LOCAL_DIR = Path("~/.local/share/bamboost").expanduser()
CACHE_DIR = Path("~/.cache/bamboost-next").expanduser()
DATABASE_FILE_NAME = "bamboost-next.sqlite"
# fmt: off
DEFAULT_EXCLUDE_DIRS: set[str] = {
    # version-control metadata
    ".git", ".hg", ".svn", ".bzr", ".cvs",

    # virtual-envs & package managers
    ".venv", "venv", "env", ".tox", ".nox",
    "node_modules", ".yarn", ".pnp",

    # build / artefact output
    "build", "dist", ".eggs",

    # IDE & editor junk
    ".idea", ".vscode", ".vs",

    # language-specific caches
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".cache", ".cargo", ".gocache",

    # web / framework caches
    ".next", ".nuxt", ".terser-cache", ".parcel-cache",
    ".vercel", ".serverless", ".aws-sam",

    # infra tools (Terraform etc.)
    ".terraform",

    # vendor copies of dependencies
    "vendor",
}
# fmt: on


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


@dataclass
class _Base:
    _field_aliases = {}

    def __repr__(self) -> str:
        s = ""
        max_length = max(len(field.name) for field in fields(self))
        for field in fields(self):
            s += f"{field.name:<{max_length + 1}}: {getattr(self, field.name)}\n"
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
    def from_dict(cls, config: dict, **kwargs) -> "Self":
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
            **kwargs: Additional keyword arguments to pass to the class constructor

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
        instance = cls(**filtered_config, **kwargs)

        # Check for missing fields (defaults used)
        for field_def in fields(cls):
            if field_def.name not in filtered_config:
                log.info(
                    f"Config key '{field_def.name}' not set; using default: {getattr(instance, field_def.name)}"
                )

        return instance


# -----------------------------
# Default configuration values
# -----------------------------


@dataclass(repr=False)
class _Paths(_Base):
    """Paths used by bamboost.

    This dataclass contains the paths used by bamboost.

    Attributes:
        localDir: The directory where the local data is stored.
        cacheDir: The directory where the cache is stored.
    """

    localDir: StrPath = field(default=LOCAL_DIR)
    cacheDir: StrPath = field(default=CACHE_DIR)

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

    log_file_lock_severity: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        field(default="WARNING")
    )
    """The severity level for the log file lock."""

    log_root_only: bool = False
    """If True, only the root logger is used."""


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

    searchPaths: Iterable[Union[str, Path]] = field(
        default_factory=lambda: PathSet([Path("~").expanduser()])
    )
    """The list of paths to search for collections."""

    excludeDirs: Iterable[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_DIRS)
    """The list of directory names to exclude from the search."""

    extendDefaultExcludeDirs: Iterable[str] | None = None
    """Use this to extend the default exclude directories."""

    syncTables: bool = field(default=True)
    """If True, the sqlite database is updated after some queries to keep it in sync."""

    convertArrays: bool = True
    """If True, sqlite lists are converted to np.arrays. If false, they are left as
    lists."""

    databaseFileName: str = field(default=DATABASE_FILE_NAME)
    """The basename of the database file."""

    databaseFile: Path = field(init=False)
    """The path to the default database file in the current context. This can be the
    global database or a project-specific database."""

    isolated: bool = False
    """If true, this project manages it's own database. The searchPaths are reduced to the
    project root only."""

    projectDir: Optional[Path] = None
    """The project directory, if found."""

    def __post_init__(self) -> None:
        # Parse search paths to Path objects
        self.searchPaths = PathSet(
            Path(p).expanduser() if isinstance(p, str) else p for p in self.searchPaths
        )

        # Handle isolated mode
        if self.isolated and self.projectDir:
            self.projectDir.joinpath(".bamboost_cache").mkdir(
                parents=True, exist_ok=True
            )
            self.databaseFile = self.projectDir.joinpath(
                ".bamboost_cache", "bamboost.sqlite"
            )
            self.searchPaths = PathSet([self.projectDir])
        else:
            # ensure that localDir exists
            makedirs(LOCAL_DIR, exist_ok=True)
            self.databaseFile = LOCAL_DIR.joinpath(self.databaseFileName)

        # Handle extendDefaultExcludeDirs
        if self.extendDefaultExcludeDirs is not None:
            self.excludeDirs = set((*self.excludeDirs, *self.extendDefaultExcludeDirs))


@dataclass(repr=False, init=False)
class _Config(_Base):
    """Configuration class for bamboost.

    This class manages the configuration options and index settings for bamboost.
    It loads the configuration from a file and provides access to the options
    and index attributes.

    Args:
        project_dir: An optional alternative directory to load the project-based config
            from.

    Attributes:
        paths: Paths used by bamboost.
        options: Configuration options for bamboost.
        index: Index settings for bamboost.
    """

    paths: _Paths
    options: _Options
    index: _IndexOptions

    def __init__(self, project_dir: Optional[StrPath] = None) -> None:
        global_config = _get_global_config(CONFIG_FILE)
        project_dir = project_dir or _find_root_dir()
        if project_dir:
            project_config = _get_project_config(Path(project_dir))
        else:
            project_config = {}

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
        self.index = _IndexOptions.from_dict(
            config.pop("index", {}), projectDir=project_dir
        )

        # Log unknown config options
        self._remainder = config
        for key in config.keys():
            log.info(f"Unknown config table: {key}")

    def __repr__(self) -> str:
        s = str()
        for field in fields(self):
            s += f"> {field.name.upper()}\n"
            s += getattr(self, field.name).__repr__()
            s += "\n"
        return s


# Create the config instance (load the configuration)
config: _Config = _Config()
