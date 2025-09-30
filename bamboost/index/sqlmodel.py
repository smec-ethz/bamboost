"""SQLAlchemy Core helper utilities for the BAMBOOST index database."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from sqlalchemy import RowMapping, Table, delete, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Insert, ReturningInsert

from bamboost import BAMBOOST_LOGGER
from bamboost.core.utilities import flatten_dict
from bamboost.index._filtering import Filter

if TYPE_CHECKING:
    from pandas import DataFrame

from .schema import collections_table, create_all, parameters_table, simulations_table

__all__ = [
    "CollectionRecord",
    "FilteredCollectionRecord",
    "ParameterRecord",
    "SimulationRecord",
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
# Dataclasses representing the loaded records
# ------------------------------------------------


@dataclass
class ParameterRecord:
    id: int
    simulation_id: int
    key: str
    value: Any


@dataclass
class SimulationRecord:
    id: int
    collection_uid: str
    name: str
    created_at: datetime
    modified_at: datetime
    description: str | None
    status: str
    submitted: bool
    parameters: list[ParameterRecord] = field(default_factory=list)

    def as_dict_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "collection_uid": self.collection_uid,
            "name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "description": self.description,
            "status": self.status,
            "submitted": self.submitted,
        }

    def parameter_dict(self) -> dict[str, Any]:
        return {parameter.key: parameter.value for parameter in self.parameters}

    def as_dict(self, standalone: bool = True) -> dict[str, Any]:
        data = {
            "collection_uid": self.collection_uid,
            "name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "description": self.description,
            "status": self.status,
            "submitted": self.submitted,
        }
        if standalone:
            data["id"] = self.id
        return data | self.parameter_dict()


@dataclass
class CollectionRecord:
    uid: str
    path: str
    created_at: datetime | None
    description: str
    tags: list[str]
    aliases: list[str]
    simulations: list[SimulationRecord] | property = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"CollectionRecord(uid={self.uid!r}, path={self.path!r}, simulations={len(self.simulations)})"

    def _repr_html_(self) -> str:  # pragma: no cover - Jupyter helper
        return f"Collection <b>{self.uid}</b><br/>Location: <a href={self.path}><i>{self.path}</i></a>"

    @property
    def parameters(self) -> list[ParameterRecord]:
        return [
            parameter
            for simulation in self.simulations
            for parameter in simulation.parameters
        ]

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


class FilteredCollectionRecord(CollectionRecord):
    def __init__(self, base: CollectionRecord, filter_: Filter) -> None:
        self._base = base
        self._filter = filter_

    @property
    def simulations(self) -> list[SimulationRecord]:
        df = self.to_pandas()
        return [
            simulation
            for simulation in self._base.simulations
            if simulation.name in df["name"].values
        ]

    def to_pandas(self) -> "DataFrame":
        df = super().to_pandas()
        return self._filter.apply(df)  # pyright: ignore[reportReturnType]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"FilteredCollection {self.uid} with {len(self.simulations)} simulations"

    def _repr_html_(self) -> str:  # pragma: no cover - Jupyter helper
        return f"<b>FilteredCollection {self.uid}</b><br/>{len(self.simulations)} simulations"


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
    parameters = _fetch_parameters_for(session, [row["id"]])
    return _build_simulation(row, parameters.values())  # pyright: ignore[reportArgumentType]


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
