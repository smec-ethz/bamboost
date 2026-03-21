"""SQLAlchemy Core schema definitions and helper utilities for the BAMBOOST index database."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    MetaData,
    RowMapping,
    String,
    Table,
    UniqueConstraint,
    delete,
    select,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Delete, Insert, ReturningInsert
from typing_extensions import Self

from bamboost import constants
from bamboost._logger import BAMBOOST_LOGGER
from bamboost.constants import (
    TABLENAME_COLLECTIONS,
    TABLENAME_PARAMETERS,
    TABLENAME_SIMULATION_LINKS,
    TABLENAME_SIMULATIONS,
)
from bamboost.core.utilities import flatten_dict
from bamboost.index._filtering import Filter, Sorter

if TYPE_CHECKING:
    from pandas import DataFrame

log = BAMBOOST_LOGGER.getChild(__name__)

# ------------------------------------------------
# Dataclasses representing the loaded records
# ------------------------------------------------


@dataclass(frozen=True)
class ParameterRecord:
    id: int
    simulation_id: int
    key: str
    value: Any


@dataclass(frozen=True)
class SimulationRecord:
    id: int
    collection_uid: str
    name: str
    created_at: datetime
    modified_at: datetime
    description: str | None
    status: str
    submitted: bool
    tags: list[str] = field(default_factory=list)
    parameters: list[ParameterRecord] = field(default_factory=list)

    links: dict[str, str] = field(default_factory=dict)

    def as_dict_metadata(self) -> dict[str, Any]:
        selected_fields = (
            "id",
            "collection_uid",
            "name",
            "created_at",
            "modified_at",
            "description",
            "tags",
            "status",
            "submitted",
        )
        return {field: getattr(self, field) for field in selected_fields}

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return {parameter.key: parameter.value for parameter in self.parameters}

    def as_dict(self, standalone: bool = True) -> dict[str, Any]:
        output_fields = (
            "name",
            "created_at",
            "description",
            "tags",
            "status",
            "submitted",
            "links",
        )
        if standalone:
            output_fields += ("id", "collection_uid", "modified_at")

        data = {field: getattr(self, field) for field in output_fields}
        return data | self.parameter_dict


@dataclass(frozen=False)
class CollectionMetadata:
    """Collection metadata container. Can load/save from/to dicts, storing unknown
    fields in an extras dictionary.
    """

    uid: str
    """Unique identifier of the collection."""
    created_at: datetime | None = field(default=None)
    """Creation timestamp."""
    tags: list[str] = field(default_factory=list)
    """List of tags associated with the collection."""
    aliases: list[str] = field(default_factory=list)
    """List of aliases for the collection."""
    author: str | dict | None = field(default=None)
    """Author information, can be a string or a dictionary."""
    description: str | None = field(default=None)
    """Completely optional description of the collection."""

    _extras: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._fields = {f.name for f in fields(self) if f.init}  # ignore _extras

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self._fields}
        d = {k: v for k, v in d.items() if v is not None}  # remove None values
        d.update(self._extras)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *_) -> Self:
        """Create an instance from a dictionary, storing unknown fields in the extras
        dictionary.

        Args:
            data: The input dictionary.
        """
        _fields = {f.name for f in fields(cls) if f.init}  # ignore _extras
        known = {k: v for k, v in data.items() if k in _fields}
        extras = {k: v for k, v in data.items() if k not in _fields}

        obj = cls(**known)
        obj._extras = extras
        return obj


@dataclass(frozen=False)
class CollectionRecord(CollectionMetadata):
    """Collection record, including metadata and associated simulations. Used for exposing
    (read only) a record in the collection table in the sql database.
    """

    path: str = field(
        default=""
    )  # default to empty string for compatibility, it should always be set
    """Path to the collection on disk. Guaranteed to be not null."""
    simulations: list[SimulationRecord] | property = field(
        default_factory=list, repr=False
    )
    """List of simulations associated with the collection."""

    @property
    def parameters(self) -> list[ParameterRecord]:
        return [
            parameter
            for simulation in self.simulations
            for parameter in simulation.parameters
        ]

    @property
    def links(self) -> dict[str, dict[str, str]]:
        """Dictionary of links for each simulation in the collection."""
        return {sim.name: sim.links for sim in self.simulations if sim.links}

    def get_parameter_keys(self) -> tuple[list[str], list[int]]:
        unique = list({parameter.key for parameter in self.parameters})
        counts = [
            sum(1 for parameter in self.parameters if parameter.key == key)
            for key in unique
        ]
        return unique, counts

    def to_pandas(self, flatten: bool = True) -> "DataFrame":
        import pandas as pd

        records = [
            simulation.as_dict(standalone=False) for simulation in self.simulations
        ]
        if flatten:
            records = [flatten_dict(record) for record in records]
        return pd.DataFrame.from_records(records)

    def filtered(self, filter: Filter | None, sorter: Sorter | None) -> Self:
        # TODO: this method is not efficient!
        if (filter is None) and (sorter is None):
            return self

        df = self.to_pandas()
        if filter is not None:
            df = filter.apply(df)
        if sorter is not None:
            tmp_map = {
                sim.name: sim
                for sim in self.simulations
                if sim.name in df["name"].values
            }
            df = sorter.apply(df)  # type: ignore
            simulations = [tmp_map[name] for name in df["name"].values]
        else:
            simulations = [
                sim for sim in self.simulations if sim.name in df["name"].values
            ]

        return dataclasses.replace(self, simulations=simulations)


# ------------------------------------------------
# SQL table definitions
# ------------------------------------------------
metadata = MetaData()

collections_table = Table(
    TABLENAME_COLLECTIONS,
    metadata,
    Column("uid", String, primary_key=True),
    Column("path", String, nullable=False),
    Column("created_at", DateTime, nullable=True),
    Column("description", String, nullable=False, default="", server_default=""),
    Column("tags", JSON, nullable=False, default=list, server_default="[]"),
    Column("aliases", JSON, nullable=False, default=list, server_default="[]"),
    Column("author", JSON, nullable=True, default=None, server_default="null"),
)

simulations_table = Table(
    TABLENAME_SIMULATIONS,
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "collection_uid",
        String,
        ForeignKey(f"{TABLENAME_COLLECTIONS}.uid", ondelete="CASCADE"),
    ),
    Column("name", String, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.now),
    Column("modified_at", DateTime, nullable=False, default=datetime.now),
    Column("description", String, nullable=True),
    Column("tags", JSON, nullable=False, default=list, server_default="[]"),
    Column(
        "status",
        String,
        nullable=False,
        default="initialized",
        server_default="initialized",
    ),
    Column("submitted", Boolean, nullable=False, default=False, server_default="0"),
    UniqueConstraint("collection_uid", "name", name="uix_collection_name"),
)

simulation_links_table = Table(
    TABLENAME_SIMULATION_LINKS,
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "source_id",
        Integer,
        ForeignKey(f"{TABLENAME_SIMULATIONS}.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "target_id",
        Integer,
        ForeignKey(f"{TABLENAME_SIMULATIONS}.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("name", String, nullable=False),
    UniqueConstraint("source_id", "name", name="uix_source_link_name"),
)

parameters_table = Table(
    TABLENAME_PARAMETERS,
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "simulation_id",
        Integer,
        ForeignKey(f"{TABLENAME_SIMULATIONS}.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("key", String, nullable=False),
    Column("value", JSON, nullable=False),
    UniqueConstraint("simulation_id", "key", name="uix_simulation_key"),
)


def create_all(engine: Engine) -> None:
    """Create all tables defined in this module."""

    metadata.create_all(engine, checkfirst=True)
    with engine.begin() as connection:
        for table in (
            collections_table,
            simulations_table,
            parameters_table,
            simulation_links_table,
        ):
            existing_columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    f"PRAGMA table_info({table.name})"
                )
            }

            for column in table.columns:
                if column.name in existing_columns:
                    continue
                connection.exec_driver_sql(
                    f"ALTER TABLE {table.name} ADD COLUMN "
                    f"{_column_sql_for_alter(engine, column)}"
                )


def _column_sql_for_alter(engine: Engine, column: Column[Any]) -> str:
    """Build a SQLite-safe column definition for ALTER TABLE ADD COLUMN."""
    type_sql = column.type.compile(dialect=engine.dialect)
    clauses = [column.name, str(type_sql)]

    default_sql = _server_default_sql(column)
    # SQLite refuses ALTER TABLE ADD COLUMN for NOT NULL columns without a default.
    # Keep migration permissive for legacy schemas by only enforcing NOT NULL when
    # a server default is available.
    if not column.nullable and default_sql is not None:
        clauses.append("NOT NULL")

    if default_sql is not None:
        clauses.append(f"DEFAULT {default_sql}")

    return " ".join(clauses)


def _server_default_sql(column: Column[Any]) -> str | None:
    if column.server_default is None:
        return None

    arg = column.server_default.arg  # type: ignore[unresolved-attribute]
    if hasattr(arg, "text"):
        return str(arg.text)
    if isinstance(arg, bool):
        return "1" if arg else "0"
    if isinstance(arg, (int, float)):
        return str(arg)
    if isinstance(arg, str):
        escaped = arg.replace("'", "''")
        return f"'{escaped}'"
    return None


# ------------------------------------------------
# JSON helpers retained for engine configuration
# ------------------------------------------------

_ENCODERS: dict[type, Callable[[Any], Any]] = {
    datetime: lambda obj: obj.isoformat(),
    complex: lambda obj: {"real": obj.real, "imag": obj.imag},
}

_DECODERS: dict[type, Callable[[Any], Any]] = {
    datetime: datetime.fromisoformat,
    complex: lambda payload: complex(payload["real"], payload["imag"]),
}


def json_serializer(value: Any) -> str:
    """Convert a value to a JSON string."""

    return json.dumps(value, cls=SqliteJSONEncoder)


def json_deserializer(value: str) -> Any:
    """Convert a JSON string to a Python value."""

    payload = json.loads(value)

    if isinstance(payload, dict) and "__type__" in payload and "__value__" in payload:
        type_name = payload["__type__"]
        converter = next(
            (decoder for t, decoder in _DECODERS.items() if t.__name__ == type_name),
            None,
        )
        if converter is not None:
            return converter(payload["__value__"])
    return payload


class SqliteJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and datetime types."""

    def default(self, obj: Any) -> Any:  # type: ignore[invalid-method-override]
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic, np.number)):
            return obj.item()

        for typ, encoder in _ENCODERS.items():
            if isinstance(obj, typ):
                return {"__type__": typ.__name__, "__value__": encoder(obj)}

        try:
            return super().default(obj)
        except TypeError:
            return f"{obj!s} (unserializable)"


# ------------------------------------------------
# Builders for dataclasses from database rows
# ------------------------------------------------


def _build_collection(session: Session, row: RowMapping) -> CollectionRecord:
    collection_uid = row["uid"]
    simulations_rows = (
        session.execute(
            select(simulations_table).where(
                simulations_table.c.collection_uid == collection_uid
            )
        )
        .mappings()
        .all()
    )

    simulation_ids = [sim_row["id"] for sim_row in simulations_rows]
    parameters_map = _fetch_parameters_for(session, simulation_ids)
    links_map = _fetch_links_for(session, simulation_ids)
    simulations = [
        _build_simulation(
            sim_row,
            parameters_map.get(sim_row["id"], []),
            links_map.get(sim_row["id"], {}),
        )
        for sim_row in simulations_rows
    ]

    return CollectionRecord(
        uid=collection_uid,
        path=row["path"],
        created_at=row.get("created_at"),
        description=row.get("description", ""),
        tags=list(row.get("tags") or []),
        aliases=list(row.get("aliases") or []),
        author=row.get("author"),
        simulations=simulations,
    )


def _build_simulation(
    row: RowMapping,
    parameters: Sequence[ParameterRecord] | None,
    links: dict[str, str] | None = None,
) -> SimulationRecord:
    return SimulationRecord(
        id=row["id"],
        collection_uid=row["collection_uid"],
        name=row["name"],
        created_at=row["created_at"],
        modified_at=row["modified_at"],
        description=row.get("description"),
        tags=list(row.get("tags") or []),
        status=row.get("status", ""),
        submitted=bool(row.get("submitted", False)),
        parameters=list(parameters or []),
        links=links or {},
    )


def _build_parameter(row: RowMapping) -> ParameterRecord:
    return ParameterRecord(
        id=row["id"],
        simulation_id=row["simulation_id"],
        key=row["key"],
        value=row["value"],
    )


def _fetch_parameters_for(
    session: Session, simulation_ids: Sequence[int]
) -> dict[int, list[ParameterRecord]]:
    if not simulation_ids:
        return {}
    rows = (
        session.execute(
            select(parameters_table).where(
                parameters_table.c.simulation_id.in_(simulation_ids)
            )
        )
        .mappings()
        .all()
    )
    result: dict[int, list[ParameterRecord]] = {}
    for row in rows:
        parameter = _build_parameter(row)
        result.setdefault(parameter.simulation_id, []).append(parameter)
    return result


def _fetch_links_for(
    session: Session, simulation_ids: Sequence[int]
) -> dict[int, dict[str, str]]:
    if not simulation_ids:
        return {}

    # We join with simulations_table to get the target UID (collection_uid:name)
    stmt = (
        select(
            simulation_links_table.c.source_id,
            simulation_links_table.c.name.label("link_name"),
            simulations_table.c.collection_uid,
            simulations_table.c.name.label("target_name"),
        )
        .select_from(
            simulation_links_table.join(
                simulations_table,
                simulation_links_table.c.target_id == simulations_table.c.id,
            )
        )
        .where(simulation_links_table.c.source_id.in_(simulation_ids))
    )
    rows = session.execute(stmt).mappings().all()

    result: dict[int, dict[str, str]] = {}

    for row in rows:
        source_id = row["source_id"]
        link_name = row["link_name"]
        target_uid = (
            f"{row['collection_uid']}{constants.UID_SEPARATOR}{row['target_name']}"
        )
        result.setdefault(source_id, {})[link_name] = target_uid
    return result


# ------------------------------------------------
# Upsert statement factories
# ------------------------------------------------


def collections_upsert_stmt(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> Insert:
    """Create an upsert statement for collections.

    Args:
        data: A single dictionary or a sequence of dictionaries representing the
            collections to be inserted or updated.
    """
    payload, _ = _normalize_payload(data, collections_table)
    update_columns = _collect_update_columns(payload, {"uid"})

    stmt = insert(collections_table).values(payload)
    set_clause = {column: getattr(stmt.excluded, column) for column in update_columns}
    return stmt.on_conflict_do_update(
        index_elements=[collections_table.c.uid], set_=set_clause
    )


def simulations_upsert_stmt(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> ReturningInsert[Any]:
    """Create an upsert statement for simulations.

    Args:
        data: A single dictionary or a sequence of dictionaries representing the
            simulations to be inserted or updated.

    Returns:
        A SQLAlchemy Insert statement with a RETURNING clause for the simulation IDs.
    """
    payload, _ = _normalize_payload(data, simulations_table)
    update_columns = _collect_update_columns(payload, {"id"})

    stmt = insert(simulations_table).values(payload)
    set_clause = {column: getattr(stmt.excluded, column) for column in update_columns}
    stmt = stmt.on_conflict_do_update(
        index_elements=[simulations_table.c.collection_uid, simulations_table.c.name],
        set_=set_clause,
    )
    return stmt.returning(simulations_table.c.id)


def parameters_upsert_stmt(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> Insert:
    payload, _ = _normalize_payload(data, parameters_table)
    stmt = insert(parameters_table).values(payload)
    return stmt.on_conflict_do_update(
        index_elements=[parameters_table.c.simulation_id, parameters_table.c.key],
        set_={"value": stmt.excluded.value},
    )


def simulation_links_upsert_stmt(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> Insert:
    payload, _ = _normalize_payload(data, simulation_links_table)
    stmt = insert(simulation_links_table).values(payload)
    return stmt.on_conflict_do_update(
        index_elements=[
            simulation_links_table.c.source_id,
            simulation_links_table.c.name,
        ],
        set_={"target_id": stmt.excluded.target_id},
    )


def delete_collection_stmt(uid: str) -> Delete:
    return delete(collections_table).where(collections_table.c.uid == uid)


def delete_simulation_stmt(collection_uid: str, name: str) -> Delete:
    return delete(simulations_table).where(
        simulations_table.c.collection_uid == collection_uid,
        simulations_table.c.name == name,
    )


def delete_simulation_links_stmt(source_id: int) -> Delete:
    return delete(simulation_links_table).where(
        simulation_links_table.c.source_id == source_id
    )


def _normalize_payload(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    table: Table,
) -> tuple[Sequence[dict[str, Any]] | dict[str, Any], bool]:
    """Ensure the payload is a list of dictionaries with only valid keys for the table.

    Args:
        data: A single dictionary or a sequence of dictionaries representing the records to be
            inserted or updated.
        table: The SQLAlchemy Table object representing the target database table.

    Returns:
        A tuple containing:
            - A list of dictionaries (or a single dictionary if the input was a single record)
              with keys filtered to match the table's columns.
            - A boolean indicating whether the input was a single record (True) or
              multiple records (False).
    """
    if isinstance(data, Mapping):
        items: Sequence[Mapping[str, Any]] = [data]
        single = True
    else:
        items = list(data)
        single = False

    filtered: list[dict[str, Any]] = []
    valid_keys = set(table.c.keys())
    for record in items:
        filtered.append({key: record[key] for key in record.keys() & valid_keys})
    return (filtered[0] if single else filtered), single


def _collect_update_columns(
    payload: Sequence[dict[str, Any]] | dict[str, Any],
    excluded_keys: set[str],
) -> set[str]:
    """Collect the set of columns to be updated in an upsert operation.

    Args:
        payload: A single dictionary or a sequence of dictionaries representing the
            records to be inserted or updated.
        excluded_keys: A set of keys to exclude from the update operation (e.g., primary keys).
    """
    if isinstance(payload, dict):
        keys = set(payload.keys())
    else:
        keys = set().union(*(record.keys() for record in payload))
    return keys - excluded_keys


# ------------------------------------------------
# Fetch helpers
# ------------------------------------------------


def fetch_collection(session: Session, uid: str) -> CollectionRecord | None:
    row = (
        session.execute(select(collections_table).where(collections_table.c.uid == uid))
        .mappings()
        .first()
    )
    if row is None:
        return None
    return _build_collection(session, row)


def fetch_collections(session: Session) -> list[CollectionRecord]:
    rows = session.execute(select(collections_table)).mappings().all()
    return [_build_collection(session, row) for row in rows]


def fetch_collection_uid_by_alias(session: Session, alias: str) -> str | None:
    if not alias:
        return None
    alias_key = alias.casefold()
    rows = session.execute(
        select(collections_table.c.uid, collections_table.c.aliases)
    ).all()

    for uid, aliases in rows:
        # if direct match on uid, return it
        if uid.casefold() == alias_key:
            return uid

        if aliases and any(
            str(item).casefold() == alias_key for item in aliases if item
        ):
            return uid
    return None


def fetch_simulation_id(
    session: Session, collection_uid: str, simulation_name: str
) -> None | int:
    """Fetch the ID of a simulation given its collection UID and name."""
    return session.execute(
        select(simulations_table.c.id).where(
            simulations_table.c.collection_uid == collection_uid,
            simulations_table.c.name == simulation_name,
        )
    ).scalar()


def fetch_simulation(
    session: Session, collection_uid: str, name: str
) -> SimulationRecord | None:
    """Fetch a simulation given its collection UID and name."""
    row = (
        session.execute(
            select(simulations_table).where(
                simulations_table.c.collection_uid == collection_uid,
                simulations_table.c.name == name,
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    parameters_map = _fetch_parameters_for(session, [row["id"]])
    links_map = _fetch_links_for(session, [row["id"]])
    return _build_simulation(
        row, parameters_map.get(row["id"], []), links_map.get(row["id"], {})
    )


def fetch_simulations(session: Session) -> list[SimulationRecord]:
    """Fetch all simulations in the database."""
    rows = session.execute(select(simulations_table)).mappings().all()
    simulation_ids = [row["id"] for row in rows]
    parameters_map = _fetch_parameters_for(session, simulation_ids)
    links_map = _fetch_links_for(session, simulation_ids)
    return [
        _build_simulation(
            row, parameters_map.get(row["id"], []), links_map.get(row["id"], {})
        )
        for row in rows
    ]


def fetch_parameters(session: Session) -> list[ParameterRecord]:
    rows = session.execute(select(parameters_table)).mappings().all()
    return [_build_parameter(row) for row in rows]
