from contextlib import contextmanager
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Protocol,
    TypeVar,
)

from bamboost.mpi import MPI

if TYPE_CHECKING:
    from bamboost.mpi import Comm

_CT = TypeVar("_CT", bound=Callable)


class HasComm(Protocol):
    _comm: "Comm"


class RootProcessMeta(type):
    """A metaclass that makes classes MPI-safe by ensuring methods are only executed on
    the root process. The class implementing this metaclass must have a `_comm` attribute
    that is an MPI communicator.

    This metaclass modifies class methods to either use broadcast communication
    (if decorated with @bcast) or to only execute on the root process (rank 0).
    """

    __exclude__ = {"__init__", "__new__"}

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
            if attr_name in mcs.__exclude__:
                continue

            # unwrap staticmethod and classmethod
            if isinstance(attr_value, (staticmethod, classmethod)):
                continue

            if callable(attr_value):
                # check for @exclude decorator
                if hasattr(attr_value, "_mpi_on_all_"):
                    continue

                # wrap the remaining methods only
                attrs[attr_name] = mcs.bcast_result(attr_value)

        return super().__new__(mcs, name, bases, attrs)

    @staticmethod
    def bcast_result(func: _CT) -> _CT:
        """Decorator that ensures a method is only executed on the root process (rank 0).

        Args:
            func (callable): The method to be decorated.

        Returns:
            callable: The wrapped method that only executes on the root process.
        """

        @wraps(func)
        def wrapper(self: HasComm, *args, **kwargs):
            status = True
            result = None
            exc = None

            if self._comm.rank == 0:
                try:
                    with comm_self(self):
                        result = func(self, *args, **kwargs)
                except Exception as e:
                    status = False
                    exc = e

            # Synchronize status, result, and exceptions collectively
            broadcast_data = self._comm.bcast((status, result, exc), root=0)

            # If an exception occurred on Rank 0, raise it collectively on all ranks
            if not broadcast_data[0]:
                raise broadcast_data[2]

            return broadcast_data[1]

        return wrapper  # ty:ignore[invalid-return-type]

    @staticmethod
    def exclude(func):
        func._mpi_on_all_ = True
        return func


@contextmanager
def comm_self(instance: HasComm) -> Generator[None, None, None]:
    """Context manager to temporarily change the communicator to MPI.COMM_SELF.

    This context manager allows collective operations (which normally require
    execution across all ranks to prevent deadlocks) to be called from a single
    rank only, as MPI.COMM_SELF represents a single-rank communicator.

    Args:
        instance: An instance of a class that has a _comm attribute (MPI communicator).

    Yields:
        None
    """
    prev_comm = instance._comm
    comm_self_val = MPI.COMM_SELF

    from bamboost.mpi.serial import NullComm, SerialComm

    # If the original communicator is a real mpi4py communicator (even when globally MPI is
    # disabled, e.g., via local comm overrides), we must swap it with mpi4py's real COMM_SELF
    # rather than the serial mock COMM_SELF.
    # We check against SerialComm/NullComm first to avoid mpi4py import overhead on serial runs.
    if not isinstance(prev_comm, (SerialComm, NullComm)):
        try:
            from mpi4py import MPI as real_MPI  # ty: ignore[unresolved-import]

            if isinstance(prev_comm, real_MPI.Comm):
                comm_self_val = real_MPI.COMM_SELF
        except ImportError:
            pass

    try:
        instance._comm = comm_self_val
        yield
    finally:
        instance._comm = prev_comm
