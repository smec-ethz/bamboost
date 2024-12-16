from functools import wraps
from typing import Any, Callable, Protocol, Sequence, TypeVar, cast

from sqlalchemy import Row, select

from bamboost import config
from bamboost.core.caching.schema import EngineWithContext
from bamboost.core.mpi import MPI

# from .sql_schema import (
#     collections,
#     metadata,
#     parameters,
#     simulations,
# )
from .model import create_all, Collection, Parameter, Simulation

# Define a TypeVar for a bound method of IndexAPI
_M = TypeVar("_M", bound=Callable[..., Any])


class _DatabaseProtocol(Protocol):
    engine: EngineWithContext


def get(method: _M) -> _M:
    comm = MPI.COMM_WORLD

    @wraps(method)
    def inner(instance: _DatabaseProtocol, *args, **kwargs):
        # handle MPI
        if comm.rank == 0:
            with instance.engine.open():
                result = method(instance, *args, **kwargs)

        return comm.bcast(result, root=0)

    return cast(_M, inner)


def write(method: _M) -> _M:
    comm = MPI.COMM_WORLD

    @wraps(method)
    def inner_root(instance: _DatabaseProtocol, *args, **kwargs):
        with instance.engine.open():
            method(instance, *args, **kwargs)

    def inner_off_root(*_args, **_kwargs):
        pass

    if comm.rank == 0:
        return cast(_M, inner_root)
    else:
        return cast(_M, inner_off_root)


# class Collection:
#     def __init__(self, id: str, engine: EngineWithContext) -> None:
#         self.id = id
#         self.engine = engine
#
#     @get
#     def read_all(self) -> Sequence[Row[Any]]:
#         query = (
#             select(
#                 simulations.c.id.label("simulation_id"),
#                 simulations.c.name.label("simulation_name"),
#                 simulations.c.created_at,
#                 simulations.c.updated_at,
#                 parameters.c.key.label("parameter_key"),
#                 parameters.c.value.label("parameter_value"),
#             )
#             .select_from(
#                 simulations.join(
#                     collections, simulations.c.collection_id == collections.c.id
#                 ).outerjoin(parameters, simulations.c.id == parameters.c.simulation_id)
#             )
#             .where(collections.c.id == self.id)
#         )
#         return self.engine.conn.execute(query).fetchall()


# engine = EngineWithContext(config.index.databaseFile)
# create_all(engine.engine)
