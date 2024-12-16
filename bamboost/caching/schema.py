# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code
"""
This module provides a class to handle sqlite databases.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
)

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
    Column("modified_at", DateTime, nullable=False),
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
        index=True,
    ),
    Column("key", String, nullable=False),
    Column("value", JSON, nullable=False),
    Column("type", String),
    UniqueConstraint("simulation_id", "key", name="uix_1"),
)

create_all = _metadata.create_all
