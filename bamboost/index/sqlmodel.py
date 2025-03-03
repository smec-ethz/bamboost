"""
SQLAlchemy ORM models for the bamboost index database.

The SQL model consists of three tables:
- `collections`: Contains information about the collections, namely uids and corresponding
  paths.
- `simulations`: Contains information about the simulations, including names, statuses,
  and parameters.
- `parameters`: Contains the parameters associated with the simulations.

Simulations are linked to collections via a foreign key, and parameters are linked to
simulations via foreign keys.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Insert,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import (
    Mapped,
    declarative_base,
    mapped_column,
    relationship,
)
from sqlalchemy.sql.dml import ReturningInsert
from typing_extensions import NotRequired, TypedDict

from bamboost import BAMBOOST_LOGGER
from bamboost.constants import (
    TABLENAME_COLLECTIONS,
    TABLENAME_PARAMETERS,
    TABLENAME_SIMULATIONS,
)

if TYPE_CHECKING:
    pass


log = BAMBOOST_LOGGER.getChild(__name__)

_Base = declarative_base()
create_all = _Base.metadata.create_all


_APIMethod = TypeVar("_APIMethod", bound=Callable[..., Any])


def json_serializer(value: Any) -> str:
    """Convert a value to a JSON string.

    Args:
        value: value to convert

    Returns:
        str: JSON string
    """
    import numpy as np

    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())

    # Convert numpy scalar types to their item
    if hasattr(value, "item"):
        return json.dumps(value.item())

    return json.dumps(value)


def json_deserializer(value: str) -> Any:
    """Convert a JSON string to a value.

    Args:
        value: JSON string to convert

    Returns:
        Any: Converted value
    """
    return json.loads(value)


class CollectionORM(_Base):
    """ORM model representing a collection of simulations.

    Attributes:
        uid: Unique identifier for the collection (primary key)
        path: file system path where the collection is stored.
        simulations: Relationship to the associated simulations, with cascade delete-orphan.
    """

    __tablename__ = TABLENAME_COLLECTIONS

    uid: Mapped[str] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String)

    # Relationships
    simulations: Mapped[List[SimulationORM]] = relationship(
        "SimulationORM", back_populates="collection", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Collection {self.uid} {self.path}"

    def _repr_html_(self) -> str:
        return f"Collection <b>{self.uid}</b><br/>Location: <a href={self.path}><i>{self.path}</i></a>"

    @classmethod
    def upsert(cls, data: Sequence[Dict[str, str]] | Dict[str, str]) -> Insert:
        """Inserts or updates collection records based on the unique uid.

        Args:
            data: Data to be upserted. Can be a single dictionary or a sequence of dictionaries.

        Returns:
            SQLAlchemy insert statement with conflict resolution.
        """
        stmt = insert(cls).values(data)
        return stmt.on_conflict_do_update(["uid"], set_=dict(stmt.excluded))

    @property
    def parameters(self) -> List[ParameterORM]:
        """Retrieves all parameters associated with the collections's simulations.

        Returns:
            List of parameters belonging to simulations in the collection.
        """
        return [p for s in self.simulations for p in s.parameters]

    def get_parameter_keys(self) -> tuple[list[str], list[int]]:
        """Extracts unique parameter keys and their occurences across all simulations.

        Returns:
            A tuple containing a list of unique parameter keys and a corresponding count list.
        """
        unique_params = list(set(p.key for p in self.parameters))
        counts = [sum(p.key == k for p in self.parameters) for k in unique_params]
        return unique_params, counts


class SimulationORM(_Base):
    """ORM model representing a simulation in a collection.

    Attributes:
        id: Unique simulation ID (primary key)
        collection_uid: Foreign key linking to `bamboost.index.sqlmodel.CollectionORM`
        name: Name of the simulation
        created_at: Timestamp when the simulation was created
        modified_at: Timestamp when the simulation was last modified
        description: Optional description of the simulation
        status: Current status of the simulation
        submitted: Indicates whether the simulation has been submitted
        collection: Relationship to the parent collection
        parameters: List of associated parameters
    """

    __tablename__ = TABLENAME_SIMULATIONS
    __table_args__ = (
        UniqueConstraint("collection_uid", "name", name="uix_collection_name"),
    )

    class _dataT(TypedDict):
        collection_uid: str
        name: str
        created_at: NotRequired[datetime]
        modified_at: NotRequired[datetime]
        description: NotRequired[str]
        status: NotRequired[str]

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, unique=True)
    collection_uid: Mapped[str] = mapped_column(ForeignKey(CollectionORM.uid))
    name: Mapped[str] = mapped_column(String, nullable=False)

    # Metadata
    created_at: Mapped[DateTime] = mapped_column(
        DateTime, nullable=False, default=datetime.now
    )
    modified_at: Mapped[DateTime] = mapped_column(
        DateTime, nullable=False, default=datetime.now
    )
    description: Mapped[Optional[str]] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    submitted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    collection: Mapped[CollectionORM] = relationship(
        "CollectionORM", back_populates="simulations"
    )
    parameters: Mapped[List[ParameterORM]] = relationship(
        "ParameterORM", back_populates="simulation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Simulation {self.collection_uid}+{self.name} [id: {self.id}]"

    @classmethod
    def upsert(cls, data: Sequence[_dataT] | _dataT) -> ReturningInsert[Tuple[int]]:
        """Generate an upsert (insert or update) statement.

        Args:
            data: Data to upsert, either a single record or a sequence of records.

        Returns:
            SQLAlchemy insert statement with conflict resolution. Returning the simulation ID.
        """
        stmt = insert(cls).values(data)
        # Get keys to update. Do not touch keys that are not in the data.
        # Assumes: for the sequence case, all dictionaries have the same keys
        keys_to_update = data.keys() if isinstance(data, dict) else data[0].keys()
        stmt = stmt.on_conflict_do_update(
            ["collection_uid", "name"],
            set_={k: v for k, v in stmt.excluded.items() if k in keys_to_update},
        )
        stmt = stmt.returning(cls.id)
        return stmt

    def as_dict(self, standalone: bool = True) -> Dict[str, Any]:
        """Return the simulation as a dictionary.

        Args:
            standalone (bool, optional): If False, "id", "collection_uid", and "modified_at" are
                excluded. Defaults to True.

        Returns:
            Dictionary representation of the simulation.
        """
        excluded_columns = (
            {
                "id",
                "collection_uid",
                "modified_at",
            }
            if not standalone
            else set()
        )
        column_names = [
            c.name for c in self.__table__.columns if c.name not in excluded_columns
        ]

        return {k: getattr(self, k) for k in column_names} | self.parameter_dict

    @property
    def parameter_dict(self) -> Dict[str, Any]:
        """Return a dictionary of parameters associated with the simulation."""
        return {p.key: p.value for p in self.parameters}


class ParameterORM(_Base):
    """ORM model representing a parameter associated with a simulation.

    Attributes:
        id: Unique parameter ID (primary key)
        simulation_id: Foreign key linking to `bamboost.index.sqlmodel.SimulationORM`
        key: Parameter key (name)
        value: Parameter value (stored as JSON)
        simulation: Relationship to the parent simulation
    """

    __tablename__ = TABLENAME_PARAMETERS
    __table_args__ = (
        UniqueConstraint("simulation_id", "key", name="uix_simulation_key"),
    )

    class _dataT(TypedDict):
        simulation_id: int
        key: str
        value: Any

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    simulation_id: Mapped[int] = mapped_column(
        ForeignKey(SimulationORM.id), nullable=False
    )
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[JSON] = mapped_column(JSON, nullable=False)

    # Relationships
    simulation: Mapped[SimulationORM] = relationship(
        "SimulationORM", back_populates="parameters"
    )

    def __repr__(self) -> str:
        return f"Parameter {self.key} = {self.value} [id: {self.id}]"

    @classmethod
    def upsert(cls, data: Sequence[_dataT] | _dataT) -> Insert:
        """Generate an upsert (insert or update) statement.

        Args:
            data: Data to upsert, either a single record or a sequence of records.

        Returns:
            Insert statement with conflict resolution.
        """
        stmt = insert(cls).values(data)
        return stmt.on_conflict_do_update(
            ["simulation_id", "key"], set_=dict(value=stmt.excluded.value)
        )
