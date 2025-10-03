"""SQLAlchemy Core helper utilities for the BAMBOOST index database."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from sqlalchemy import RowMapping, Table, delete, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Insert, ReturningInsert

from bamboost import BAMBOOST_LOGGER

if TYPE_CHECKING:
    pass

from .schema import (
    CollectionRecord,
    ParameterRecord,
    SimulationRecord,
    collections_table,
    create_all,
    parameters_table,
    simulations_table,
)

__all__ = [
    "collections_table",
    "create_all",
    "collections_upsert_stmt",
    "simulations_upsert_stmt",
    "parameters_upsert_stmt",
    "fetch_collection",
    "fetch_collections",
    "fetch_collection_uid_by_alias",
    "fetch_simulation",
    "fetch_simulations",
    "fetch_parameters",
    "delete_collection_stmt",
    "delete_simulation_stmt",
    "json_serializer",
    "json_deserializer",
]

log = BAMBOOST_LOGGER.getChild(__name__)

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

    def default(self, obj: Any) -> Any:
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
    simulations = [
        _build_simulation(sim_row, parameters_map.get(sim_row["id"], []))
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
) -> SimulationRecord:
    return SimulationRecord(
        id=row["id"],
        collection_uid=row["collection_uid"],
        name=row["name"],
        created_at=row["created_at"],
        modified_at=row["modified_at"],
        description=row.get("description"),
        status=row.get("status", ""),
        submitted=bool(row.get("submitted", False)),
        parameters=list(parameters or []),
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


def fetch_simulation(
    session: Session, collection_uid: str, name: str
) -> SimulationRecord | None:
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
    return _build_simulation(row, parameters_map.get(row["id"], []))


def fetch_simulations(session: Session) -> list[SimulationRecord]:
    rows = session.execute(select(simulations_table)).mappings().all()
    simulation_ids = [row["id"] for row in rows]
    parameters_map = _fetch_parameters_for(session, simulation_ids)
    return [_build_simulation(row, parameters_map.get(row["id"], [])) for row in rows]


def fetch_parameters(session: Session) -> list[ParameterRecord]:
    rows = session.execute(select(parameters_table)).mappings().all()
    return [_build_parameter(row) for row in rows]


def delete_collection_stmt(uid: str):
    return delete(collections_table).where(collections_table.c.uid == uid)


def delete_simulation_stmt(collection_uid: str, name: str):
    return delete(simulations_table).where(
        simulations_table.c.collection_uid == collection_uid,
        simulations_table.c.name == name,
    )
