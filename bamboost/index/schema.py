"""SQLAlchemy Core schema definitions for the BAMBOOST index database."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
)
from typing_extensions import Self

from bamboost.constants import (
    TABLENAME_COLLECTIONS,
    TABLENAME_PARAMETERS,
    TABLENAME_SIMULATIONS,
)
from bamboost.core.utilities import flatten_dict
from bamboost.index._filtering import Filter, Sorter

if TYPE_CHECKING:
    from pandas import DataFrame


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

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return {parameter.key: parameter.value for parameter in self.parameters}

    def as_dict(self, standalone: bool = True) -> dict[str, Any]:
        data = {
            "name": self.name,
            "created_at": self.created_at,
            "description": self.description,
            "status": self.status,
            "submitted": self.submitted,
        }
        if standalone:
            data.update(
                {
                    "id": self.id,
                    "collection_uid": self.collection_uid,
                    "modified_at": self.modified_at,
                }
            )
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
                if sim.name in df["name"].values  # type: ignore
            }
            df = sorter.apply(df)  # type: ignore
            simulations = [tmp_map[name] for name in df["name"].values]
        else:
            simulations = [
                sim
                for sim in self.simulations
                if sim.name in df["name"].values  # type: ignore
            ]

        return dataclasses.replace(self, simulations=simulations)


# ------------------------------------------------
# SQL table definitions
# these should match the dataclasses above
# ------------------------------------------------
metadata = MetaData()


def create_all(engine) -> None:
    """Create all tables defined in this module."""

    metadata.create_all(engine, checkfirst=True)


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
