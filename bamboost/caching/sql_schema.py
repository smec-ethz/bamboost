from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
)

__all__ = [
    "_metadata",
    "collections",
    "simulations",
    "parameters",
]


_metadata = MetaData()

collections = Table(
    "Collections",
    _metadata,
    Column("id", String, primary_key=True),
    Column("path", String),
)

simulations = Table(
    "Simulations",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "collection_id",
        String,
        ForeignKey("Collections.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("name", String),
    Column("description", String),
    Column("status", String, nullable=False, default="pending"),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

parameters = Table(
    "Parameters",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "simulation_id",
        Integer,
        ForeignKey("Simulations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("key", String, nullable=False),
    Column("value", JSON, nullable=False),
    Column("type", String),
)

create_all = _metadata.create_all
