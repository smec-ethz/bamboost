"""SQLAlchemy Core schema definitions for the BAMBOOST index database."""

from __future__ import annotations

from datetime import datetime

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

from bamboost.constants import (
    TABLENAME_COLLECTIONS,
    TABLENAME_PARAMETERS,
    TABLENAME_SIMULATIONS,
)

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


def create_all(engine) -> None:
    """Create all tables defined in this module."""

    metadata.create_all(engine, checkfirst=True)
