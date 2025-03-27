import sys
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Callable, Generator, Protocol, cast

from bamboost._typing import _P, _T
from bamboost.mpi import MPI

if TYPE_CHECKING:
    from bamboost.mpi import Comm


def on_rank(func: Callable[_P, _T], comm: "Comm", rank: int) -> Callable[_P, _T]:
    """Decorator to run a function on a specific rank and broadcast the result.

    Args:
        func: The function to decorate.
        comm: The MPI communicator.
        rank: The rank to run the function on.
    """

    @wraps(func)
    def inner(*args, **kwargs) -> _T:
        result = None
        if comm.rank == rank:
            result = func(*args, **kwargs)
        return cast(_T, comm.bcast(result, root=rank))

    return inner


def on_root(func: Callable[_P, _T], comm: "Comm") -> Callable[_P, _T]:
    """Decorator to run a function on the root rank and broadcast the result.

    Args:
        func: The function to decorate.
        comm: The MPI communicator.
    """
    return on_rank(func, comm, 0)


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
            if (
                callable(attr_value)
                # and not attr_name.startswith("__")
                and attr_name not in mcs.__exclude__
            ):
                if hasattr(
                    attr_value, "_mpi_bcast_"
                ):  # check for @cast_result decorator
                    continue
                if hasattr(attr_value, "_mpi_on_all_"):  # check for @exclude decorator
                    continue
                else:
                    attrs[attr_name] = mcs._root_only_default(attr_value)
        return super().__new__(mcs, name, bases, attrs)

    @staticmethod
    def _root_only_default(func):
        """Decorator that ensures a method is only executed on the root process (rank 0).

        Args:
            func (callable): The method to be decorated.

        Returns:
            callable: The wrapped method that only executes on the root process.
        """

        @wraps(func)
        def wrapper(self: HasComm, *args, **kwargs):
            result = None

            if self._comm.rank == 0:
                with RootProcessMeta.comm_self(self):
                    result = func(self, *args, **kwargs)

            # dummy broadcast to ensure synchronization
            self._comm.bcast(result, root=0)
            return result

        return wrapper

    @staticmethod
    @contextmanager
    def comm_self(instance: HasComm) -> Generator[None, None, None]:
        """Context manager to temporarily change the communicator to MPI.COMM_SELF.

        Args:
            comm: The MPI communicator.

        Yields:
            None
        """
        prev_comm = instance._comm
        try:
            instance._comm = MPI.COMM_SELF
            yield
        finally:
            instance._comm = prev_comm

    @staticmethod
    def bcast_result(func):
        @wraps(func)
        def wrapper(self: HasComm, *args, **kwargs):
            result = None

            if self._comm.rank == 0:
                with RootProcessMeta.comm_self(self):
                    result = func(self, *args, **kwargs)

            result = self._comm.bcast(result, root=0)
            return result

        wrapper._mpi_bcast_ = True  # type: ignore
        return wrapper

    @staticmethod
    def exclude(func):
        func._mpi_on_all_ = True
        return func
