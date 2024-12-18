from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from functools import reduce, wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import (
    Mapped,
    Session,
    create_session,
    declarative_base,
    joinedload,
    mapped_column,
    relationship,
)
from sqlalchemy.sql import text

from bamboost import BAMBOOST_LOGGER
from bamboost.core.mpi import MPI

if TYPE_CHECKING:
    from typing import Generator

    from mpi4py.MPI import Comm
    from sqlalchemy import Engine, Row
    from sqlalchemy.sql import ClauseElement
    from typing_extensions import TypeAlias


log = BAMBOOST_LOGGER.getChild(__name__)

_Base = declarative_base()

StrPath: TypeAlias = Union[str, Path]


class _SimulationMetadataT(TypedDict):
    created_at: datetime
    modified_at: datetime
    description: str
    status: str


_SimulationParameterT = Dict[str, Any]


_APIMethod = TypeVar("_APIMethod", bound=Callable[..., Any])


def compose_while_not_none(*funcs: Callable) -> Callable:
    """Compose multiple functions into a single function. The output of each
    function is passed as the input to the next function. The functions are
    applied from left to right. If a function returns `None`, the next function
    is not called and `None` is returned.

    Args:
        *funcs: Functions to compose.

    Returns:
        Callable: The composed function.
    """

    def function(f: Any, g: Callable) -> Callable:
        def composed(*x: Any) -> Any:
            result = f(*x)
            if result is None:
                return None
            else:
                return g(result)

        return composed

    return reduce(function, funcs)


def _json_serializer(value: Any) -> str:
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


def _json_deserializer(value: str) -> Any:
    """Convert a JSON string to a value.

    Args:
        value: JSON string to convert

    Returns:
        Any: Converted value
    """
    return json.loads(value)


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
        self.engine: Engine = create_engine(
            f"sqlite:///{file}",
            json_serializer=_json_serializer,
            json_deserializer=_json_deserializer,
        )
        self.session: Session = create_session(self.engine)

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
        try:
            self.connect(make_changes=make_changes)
            yield self.session
        finally:
            self.close()

    @overload
    def query(self, stmt: str) -> Sequence[Row[Any]]: ...

    @overload
    def query(self, stmt: ClauseElement) -> Sequence[Row[Any]]: ...

    @_bcast
    @_with_scope()
    def query(self, stmt: Any) -> Sequence[Row[Any]]:
        if isinstance(stmt, str):
            stmt = text(stmt)

        return self.session.execute(stmt).fetchall()

    @_with_scope(make_changes=True)
    def delete(self, instance: object) -> None:
        self.session.delete(instance)

    @_with_scope(make_changes=True)
    def delete_collection(self, uid: str) -> None:
        compose_while_not_none(self.session.query(Collection).get, self.session.delete)(
            uid
        )

    @_with_scope(make_changes=True)
    def delete_simulation(self, collection_id: str, simulation_name: str) -> None:
        compose_while_not_none(
            self.session.query(Simulation)
            .filter(
                Simulation.collection_uid == collection_id,
                Simulation.name == simulation_name,
            )
            .first,
            self.session.delete,
        )()

    @_bcast
    @_with_scope()
    def get_collections(self) -> List[Collection]:
        log.debug(f"Getting collections on rank {self._comm.rank}")
        return self.session.query(Collection).all()

    @_bcast
    @_with_scope()
    def get_collection(
        self, uid: Optional[str] = None, path: Optional[str] = None
    ) -> Collection:
        if uid:
            filter_stmt = Collection.uid == uid
        elif path:
            filter_stmt = Collection.path == path
        else:
            raise ValueError("Either uid or path must be provided")

        collection = (
            self.session.query(Collection)
            .options(
                joinedload(Collection.simulations).subqueryload(Simulation.parameters)
            )
            .filter(filter_stmt)
            .one()
        )
        return collection

    @_bcast
    @_with_scope()
    def get_simulation(self, collection_id: str, simulation_name: str) -> Simulation:
        return (
            self.session.query(Simulation)
            .options(joinedload(Simulation.parameters))
            .filter(
                Simulation.collection_uid == collection_id,
                Simulation.name == simulation_name,
            )
            .one()
        )

    @_with_scope(make_changes=True)
    def insert_collection(self, collection: Collection) -> None:
        self.session.add(collection)

    @_with_scope(make_changes=True)
    def insert_simulation(self, simulation: Simulation) -> None:
        self.session.add(simulation)

    @_with_scope(make_changes=True)
    def insert_parameter(self, parameter: Parameter) -> None:
        self.session.add(parameter)

    @_with_scope(make_changes=True)
    def update_collection(self, uid: str, path: str) -> None:
        stmt = (
            insert(Collection)
            .values(uid=uid, path=path)
            .on_conflict_do_update(["uid"], set_=dict(path=path))
        )
        self.session.execute(stmt)

    @_with_scope(make_changes=True)
    def update_simulation(
        self,
        collection_id: str,
        simulation_name: str,
        metadata: _SimulationMetadataT,
        params: _SimulationParameterT,
    ) -> None:
        stmt = (
            insert(Simulation)
            .values(
                collection_id=collection_id,
                name=simulation_name,
                **metadata,
            )
            .on_conflict_do_update(["collection_id", "name"], set_=metadata)
        )
        log.debug(f"Updating simulation {collection_id}+{simulation_name}")
        result = self.session.execute(stmt)

        self.update_parameters(result.lastrowid, params=params)

    @_with_scope(make_changes=True)
    def update_parameters(
        self, simulation_id: int, params: _SimulationParameterT
    ) -> None:
        for k, v in params.items():
            stmt = (
                insert(Parameter)
                .values(simulation_id=simulation_id, key=k, value=v)
                .on_conflict_do_update(["simulation_id", "key"], set_=dict(value=v))
            )
            log.debug(f"Updating parameter {k} = {v} for simulation {simulation_id}")
            self.session.execute(stmt)


class Collection(_Base):
    __tablename__ = "collections"

    uid: Mapped[str] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String)

    # Relationships
    simulations: Mapped[List[Simulation]] = relationship(
        "Simulation", back_populates="collection", cascade="all, delete-orphan"
    )

    def __init__(self, uid: str, path: str) -> None:
        self.uid = uid
        self.path = path

    def __repr__(self) -> str:
        return f"<Collection {self.uid}>"

    def as_tuple(self) -> tuple[str, str]:
        """Return the collection as a tuple of (uid, path)."""
        return self.uid, self.path


class Simulation(_Base):
    __tablename__ = "simulations"
    __table_args__ = (
        UniqueConstraint("collection_uid", "name", name="uix_collection_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, unique=True)
    collection_uid: Mapped[str] = mapped_column(ForeignKey(Collection.uid))
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
    collection: Mapped[Collection] = relationship(
        "Collection", back_populates="simulations"
    )
    parameters: Mapped[List[Parameter]] = relationship(
        "Parameter", back_populates="simulation", cascade="all, delete-orphan"
    )

    @overload
    def __init__(
        self,
        id: Optional[int],
        collection_id: str,
        name: str,
        created_at: Optional[datetime],
        modified_at: Optional[datetime],
        description: Optional[str],
        status: Optional[str],
    ) -> None: ...

    @overload
    def __init__(self, *args, **kwargs) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Simulation {self.collection_uid}+{self.name}>"

    def update_metadata(self, metadata: _SimulationMetadataT) -> None:
        for key, value in metadata.items():
            setattr(self, key, value)

    def update_parameters(self, params: _SimulationParameterT) -> None:
        self.parameters.clear()

        for key, value in params.items():
            new_parameter = Parameter(simulation_id=self.id, key=key, value=value)
            self.parameters.append(new_parameter)


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
        return f"<Parameter {self.key} = {self.value}>"
