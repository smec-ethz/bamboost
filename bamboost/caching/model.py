from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import JSON, DateTime, Engine, ForeignKey, String
from sqlalchemy.orm import (
    Mapped,
    declarative_base,
    mapped_column,
    relationship,
)

Base = declarative_base()


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String)

    # Relationships
    simulations: Mapped[List[Simulation]] = relationship(
        "Simulation", back_populates="collection", cascade="all, delete-orphan"
    )

    def __init__(self, id: str, path: str) -> None:
        self.id = id
        self.path = path

    def __repr__(self) -> str:
        return f"<Collection {self.id}>"


class Simulation(Base):
    __tablename__ = "simulations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    collection_id: Mapped[str] = mapped_column(ForeignKey(Collection.id))
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime, nullable=False, default=datetime.now
    )
    modified_at: Mapped[DateTime] = mapped_column(
        DateTime, nullable=False, default=datetime.now
    )
    description: Mapped[Optional[str]] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")

    # Relationships
    collection: Mapped[Collection] = relationship(
        "Collection", back_populates="simulations"
    )
    parameters: Mapped[List[Parameter]] = relationship(
        "Parameter", back_populates="simulation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Simulation {self.name}>"


class Parameter(Base):
    __tablename__ = "parameters"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    simulation_id: Mapped[int] = mapped_column(
        ForeignKey(Simulation.id), nullable=False
    )
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[JSON] = mapped_column(JSON, nullable=False)

    # Relationships
    simulation: Mapped[Simulation] = relationship(
        "Simulation", back_populates="parameters"
    )

    def __repr__(self) -> str:
        return f"<Parameter {self.key}={self.value}>"


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
