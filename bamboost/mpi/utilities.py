from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Protocol,
    TypeVar,
    cast,
)

from bamboost.mpi import MPI

if TYPE_CHECKING:
    from bamboost.mpi import Comm

T = TypeVar("T")


class HasComm(Protocol):
    _comm: "Comm"


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


class ParallelProxy:
    """Handles the actual MPI communication routing at runtime."""

    def __init__(self, serial_instance, comm, root: int = 0):
        self.comm = comm
        self.rank = self.comm.rank
        self.root = root
        self._core = serial_instance  # Valid object on root, None on others

    def __getattr__(self, name: str) -> Any:
        # Step 1: Check on the root process what type of attribute this actually is.
        # Workers don't have the object, so they default to assuming it's a method
        # unless told otherwise via a collective broadcast.
        is_callable = False
        if self.rank == self.root:
            if self._core is None:
                raise RuntimeError("Serial core instance missing on root process.")
            attr = getattr(self._core, name)
            is_callable = callable(attr)

        # Share whether it's a method or a property/attribute with all ranks
        is_callable = self.comm.bcast(is_callable, root=self.root)

        if is_callable:

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                status = True
                result = None
                exc = None

                if self.rank == self.root:
                    target_method = getattr(self._core, name)
                    try:
                        if hasattr(self._core, "_comm"):
                            with comm_self(self._core):
                                result = target_method(*args, **kwargs)
                        else:
                            result = target_method(*args, **kwargs)
                    except Exception as e:
                        status = False
                        exc = e

                # Synchronize status, result, and exceptions collectively
                broadcast_data = self.comm.bcast((status, result, exc), root=self.root)

                # If an exception occurred on Root, raise it collectively on all ranks
                if not broadcast_data[0]:
                    raise broadcast_data[2]

                return broadcast_data[1]

            return wrapper

        # It's a property or attribute
        # We must evaluate and sync its value immediately right here.
        else:
            if self.rank == self.root:
                result = getattr(self._core, name)
            else:
                result = None

            return self.comm.bcast(result, root=self.root)


def parallel_proxy(
    serial_class: type[T] | Callable[..., T], comm, root: int = 0, *args, **kwargs
) -> T:
    """
    Instantiates the serial class on the root process and wraps it in a proxy.
    Tells type checkers that the returned object is an instance of `T` (not Proxy).
    """
    rank = comm.rank

    # Instantiate the backend only on the designated root rank
    if rank == root:
        instance = serial_class(*args, **kwargs)
    else:
        instance = None

    proxy = ParallelProxy(instance, root=root, comm=comm)

    return cast(T, proxy)
