from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

from sqlalchemy import (
    JSON,
    DateTime,
    Engine,
    ForeignKey,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    create_session,
    declarative_base,
    joinedload,
    lazyload,
    mapped_column,
    relationship,
)
from typing_extensions import TypeAlias

from bamboost import BAMBOOST_LOGGER
from bamboost.core.mpi import MPI

if TYPE_CHECKING:
    from mpi4py.MPI import Comm


log = BAMBOOST_LOGGER.getChild(__name__)

_Base = declarative_base()

StrPath: TypeAlias = Union[str, Path]

_APIMethod = TypeVar("_APIMethod", bound=Callable[..., Any])


def _json_serializer(value: Any) -> str:
    """Convert a value to a JSON string.

    Args:
        value: value to convert

    Returns:
        str: JSON string
    """
    import json

    if hasattr(value, "item"):
        return json.dumps(value.item())

    return json.dumps(value)


comm = MPI.COMM_WORLD


class MPISafeMeta(type):
    """A metaclass that makes classes MPI-safe by ensuring methods are only
    executed on the root process.

    This metaclass modifies class methods to either use broadcast communication
    (if decorated with @bcast) or to only execute on the root process (rank 0).
    """

    def __new__(mcs, name: str, bases: tuple, attrs: dict):
        """Create a new class with MPI-safe methods.

        Args:
            name: The name of the class being created.
            bases: The base classes of the class being created.
            attrs: The attributes of the class being created.

        Returns:
            type: The new class with MPI-safe methods.
        """
        for attr_name, attr_value in attrs.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                if hasattr(attr_value, "_bcast"):
                    attrs[attr_name] = attr_value
                else:
                    attrs[attr_name] = mcs.root_only(attr_value)
        return super().__new__(mcs, name, bases, attrs)

    @staticmethod
    def root_only(func):
        """Decorator that ensures a method is only executed on the root process
        (rank 0).

        Args:
            func (callable): The method to be decorated.

        Returns:
            callable: The wrapped method that only executes on the root process.
        """

        @wraps(func)
        def wrapper(self: CacheAPI, *args, **kwargs):
            if self._comm.rank == 0:
                return func(self, *args, **kwargs)

        return wrapper


def _bcast(func):
    @wraps(func)
    def wrapper(self: CacheAPI, *args, **kwargs):
        result = None
        if self._comm.rank == 0:
            result = func(self, *args, **kwargs)
        return self._comm.bcast(result, root=0)

    wrapper._bcast = True  # type: ignore
    return wrapper


def _with_scope(make_changes: bool = False):
    """Decorator that provides a scoped transaction to a method.

    It opens a connection to the database and handles commits if
    `make_changes=True`.

    Args:
        make_changes (bool, optional): Whether the method will make changes to
            the database.

    Returns:
        callable: The wrapped method that has a session provided.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self: CacheAPI, *args, **kwargs):
            with self.scoped_session(make_changes=make_changes):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


class CacheAPI(metaclass=MPISafeMeta):
    """API for interacting with the cache database.

    The metaclass ensures that methods are only executed on the root process.
    If they are decorated with `_bcast`, their results are broadcast to all
    other processes.

    Args:
        file: The path to the database file.
        comm: The MPI communicator to use. Defaults to `MPI.COMM_WORLD`.
    """

    def __init__(self, file: StrPath, *, comm: Optional[Comm] = None) -> None:
        self.file = file
        self._comm = comm or MPI.COMM_WORLD
        self._init_engine(file)

    def _init_engine(self, file: StrPath) -> None:
        self.engine: Engine = create_engine(f"sqlite:///{file}")
        self.session: Session = create_session(self.engine)
        print(f"CacheAPI init from rank {self._comm.rank}")

        # Create all tables
        _Base.metadata.create_all(self.engine)

        self._context_stack: int = 0
        self._needs_commit: bool = False

    def connect(self, make_changes: bool = False) -> None:
        self._context_stack += 1
        self._needs_commit = self._needs_commit or make_changes
        self.session.connection()

    def close(self) -> None:
        self._context_stack -= 1
        if self._context_stack <= 0:
            if self._needs_commit:
                self.session.commit()
            self.session.close()

    @contextmanager
    def scoped_session(
        self, make_changes: bool = False
    ) -> Generator[Session, None, None]:
        print(f"Transaction from rank {self._comm.rank}")
        try:
            self.connect(make_changes=make_changes)
            yield self.session
        finally:
            self.close()

    @_bcast
    @_with_scope()
    def get_collections(self) -> List[Collection]:
        log.info(f"Getting collections on rank {self._comm.rank}")
        return self.session.query(Collection).all()

    @_bcast
    @_with_scope()
    def get_collection(self, uid: str) -> Collection:
        collection = (
            self.session.query(Collection)
            .options(
                joinedload(Collection.simulations).subqueryload(Simulation.parameters)
            )
            .filter(Collection.id == uid)
            .scalar()
        )
        return collection


class Collection(_Base):
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

    def as_tuple(self) -> tuple[str, str]:
        return self.id, self.path


class Simulation(_Base):
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


class Parameter(_Base):
    __tablename__ = "parameters"
    __table_args__ = (
        UniqueConstraint("simulation_id", "key", name="uix_simulation_key"),
    )

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
