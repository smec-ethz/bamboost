from functools import wraps
from typing import TYPE_CHECKING, Callable, Protocol, cast

from bamboost._typing import _P, _T

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


class ClassWithComm(Protocol):
    _comm: "Comm"


def bcast(func):
    @wraps(func)
    def wrapper(self: ClassWithComm, *args, **kwargs):
        result = None
        if self._comm.rank == 0:
            result = func(self, *args, **kwargs)
        return self._comm.bcast(result, root=0)

    wrapper._bcast = True  # type: ignore
    return wrapper


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
            if (
                callable(attr_value)
                and not attr_name.startswith("__")
                and attr_name != "sql_transaction"
            ):
                if hasattr(attr_value, "_bcast"):  # check for @bcast decorator
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
        def wrapper(self: ClassWithComm, *args, **kwargs):
            if self._comm.rank == 0:
                return func(self, *args, **kwargs)

        return wrapper
