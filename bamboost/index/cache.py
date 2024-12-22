from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
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
    Union,
)

from sqlalchemy import (
    JSON,
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
from typing_extensions import NotRequired, ParamSpec, TypedDict

from bamboost import BAMBOOST_LOGGER

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


log = BAMBOOST_LOGGER.getChild(__name__)

_Base = declarative_base()
create_all = _Base.metadata.create_all


class _SimulationMetadataT(TypedDict):
    created_at: datetime
    modified_at: datetime
    description: str
    status: str


_SimulationParameterT = Dict[str, Any]
_APIMethod = TypeVar("_APIMethod", bound=Callable[..., Any])
_T = TypeVar("_T")
_U = TypeVar("_U")
_P = ParamSpec("_P")


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
    __tablename__ = "collections"

    uid: Mapped[str] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String)

    # Relationships
    simulations: Mapped[List[SimulationORM]] = relationship(
        "SimulationORM", back_populates="collection", cascade="all, delete-orphan"
    )

    def __init__(self, uid: str, path: str) -> None:
        self.uid = uid
        self.path = path

    def __repr__(self) -> str:
        return f"<Collection {self.uid} {self.path}>"

    def as_tuple(self) -> tuple[str, str]:
        """Return the collection as a tuple of (uid, path)."""
        return self.uid, self.path

    @classmethod
    def upsert(cls, data: Sequence[Dict[str, str]] | Dict[str, str]) -> Insert:
        stmt = insert(cls).values(data)
        return stmt.on_conflict_do_update(["uid"], set_=dict(stmt.excluded))


class SimulationORM(_Base):
    __tablename__ = "simulations"
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

    # Relationships
    collection: Mapped[CollectionORM] = relationship(
        "CollectionORM", back_populates="simulations"
    )
    parameters: Mapped[List[ParameterORM]] = relationship(
        "ParameterORM", back_populates="simulation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Simulation {self.collection_uid}+{self.name}>"

    @classmethod
    def upsert(cls, data: Sequence[_dataT] | _dataT) -> ReturningInsert[Tuple[int]]:
        stmt = insert(cls).values(data)
        stmt = stmt.on_conflict_do_update(
            ["collection_uid", "name"],
            set_={k: v for k, v in stmt.excluded.items() if k != "id"},
        )
        stmt = stmt.returning(cls.id)
        return stmt


class ParameterORM(_Base):
    __tablename__ = "parameters"
    __table_args__ = (
        UniqueConstraint("simulation_id", "key", name="uix_simulation_key"),
    )
    _dataT = TypedDict("_dataT", simulation_id=int, key=str, value=Any)

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
        return f"<Parameter {self.key} = {self.value}>"

    @classmethod
    def upsert(cls, data: Sequence[_dataT] | _dataT) -> Insert:
        stmt = insert(cls).values(data)
        return stmt.on_conflict_do_update(
            ["simulation_id", "key"], set_=dict(value=stmt.excluded.value)
        )
