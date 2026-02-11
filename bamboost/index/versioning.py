from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path


class Version(Enum):
    V0_10 = ("0.10", "bamboost-next.sqlite")
    V0_11 = ("0.11", "bamboost.0.11.sqlite")

    def __init__(self, version: str, database_file_name: str):
        self.version = version
        self.database_file_name = database_file_name

    @classmethod
    def latest(cls) -> "Version":
        """Return the latest version."""
        return max(cls, key=lambda v: tuple(map(int, v.version.split("."))))

    @classmethod
    def from_str(cls, version_str: str) -> "Version":
        """Get enum member by version string."""
        for v in cls:
            if v.version == version_str:
                return v
        raise ValueError(f"{version_str!r} is not a valid {cls.__name__}")

    def __str__(self) -> str:
        return self.version


# ----------------------------------------
# A few Exceptions raised by this module.
# ----------------------------------------
class MigrationError(RuntimeError):
    """Base class for migration-related errors."""


class MigrationNotAvailableError(MigrationError):
    """Raised when no migration routine is registered for a version pair."""

    def __init__(self, from_version: Version, to_version: Version) -> None:
        super().__init__(
            f"No migration registered from {from_version.value.version} to {to_version.value.version}"
        )


class SourceDatabaseNotFoundError(MigrationError):
    """Raised when the source database file cannot be located."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        super().__init__(f"Database file not found at {source_path}")


# ----------------------------------------
# Migration registry and related functions.
# ----------------------------------------
MigrationExecutor = Callable[[Path, Path, bool | None], None]
_MIGRATIONS: dict[tuple[Version, Version], MigrationExecutor] = {}


def _register_migration(
    from_version: Version, to_version: Version
) -> Callable[[MigrationExecutor], MigrationExecutor]:
    """Decorator to register a migration executor for a version pair."""

    def decorator(executor: MigrationExecutor) -> MigrationExecutor:
        if (from_version, to_version) in _MIGRATIONS:
            raise RuntimeError(
                f"Migration from {from_version} to {to_version} already registered."
            )
        _MIGRATIONS[(from_version, to_version)] = executor
        return executor

    return decorator


def _get_migration(from_version: Version, to_version: Version) -> MigrationExecutor:
    """Return the migration function or raise."""
    try:
        return _MIGRATIONS[(from_version, to_version)]
    except KeyError:
        raise MigrationNotAvailableError(from_version, to_version)


def migrate_database(
    from_version: Version,
    to_version: Version,
    *,
    source_db: Path | None = None,
    destination_db: Path | None = None,
    update: bool = False,
) -> None:
    """Migrate the index database between two schema versions.

    Args:
        from_version: The source version.
        to_version: The destination version.
        source_db: Optional path to the source database file. If not provided,
            it will be determined from the config.
        destination_db: Optional path to the destination database file. If not provided,
            it will be determined from the config.
        update: If True, existing records in the destination will be updated
            with values from the source. Otherwise, existing records will be left
            unchanged.
    """
    migration = _get_migration(from_version, to_version)

    from bamboost import config
    from bamboost.index import Index

    source_db = source_db or (
        Path(config.paths.localDir) / from_version.database_file_name
    )
    destination_db = destination_db or Path(config.index.databaseFile)

    if not source_db.exists():
        raise SourceDatabaseNotFoundError(source_db)

    # Ensure the destination database exists prior to migration.
    Index()

    migration(source_db, destination_db, update=update)


# ----------------------------------------
# Migration implementations.
# ----------------------------------------
@_register_migration(Version.V0_10, Version.V0_11)
def _migrate_v0_10_to_v0_11(
    source_db: Path, destination_db: Path, *, update: bool = False
) -> None:
    """Migrate database content from version 0.10 to 0.11.

    If `update=True`, existing records in the destination are updated
    with values from the source.
    """
    import sqlite3

    tables: dict[str, dict] = {
        "collections": {
            "select": "SELECT uid, path FROM collections",
            "columns": ["uid", "path", "description", "tags", "aliases", "author"],
            "defaults": ["", "[]", "[]", "null"],  # new columns with static defaults
            "conflict": "uid",
            "update_cols": ["path"],
        },
        "simulations": {
            "select": """
                SELECT id, collection_uid, name, created_at, modified_at,
                       description, status, submitted
                FROM simulations
            """,
            "columns": [
                "id",
                "collection_uid",
                "name",
                "created_at",
                "modified_at",
                "description",
                "tags",
                "status",
                "submitted",
            ],
            "defaults": ["[]"],
            "conflict": "id",
            "update_cols": [
                "collection_uid",
                "name",
                "created_at",
                "modified_at",
                "description",
                "tags",
                "status",
                "submitted",
            ],
        },
        "parameters": {
            "select": "SELECT id, simulation_id, key, value FROM parameters",
            "columns": ["id", "simulation_id", "key", "value"],
            "conflict": "id",
            "update_cols": ["simulation_id", "key", "value"],
        },
    }

    with (
        sqlite3.connect(source_db) as src,
        sqlite3.connect(destination_db) as dst,
    ):
        src_cur, dst_cur = src.cursor(), dst.cursor()

        for table, spec in tables.items():
            src_cur.execute(spec["select"])
            rows = src_cur.fetchall()
            if not rows:
                continue

            cols = spec["columns"]
            placeholders = ", ".join("?" for _ in cols)

            if "defaults" in spec:
                # append static defaults if present
                values = [tuple(r + tuple(spec["defaults"])) for r in rows]
            else:
                values = rows

            insert_cols = ", ".join(spec["columns"])
            if update:
                set_clause = ", ".join(f"{c}=excluded.{c}" for c in spec["update_cols"])
                sql = f"""
                    INSERT INTO {table} ({insert_cols})
                    VALUES ({placeholders})
                    ON CONFLICT({spec["conflict"]}) DO UPDATE SET {set_clause}
                """
            else:
                sql = f"""
                    INSERT OR IGNORE INTO {table} ({insert_cols})
                    VALUES ({placeholders})
                """

            dst_cur.executemany(sql, values)

        dst.commit()
