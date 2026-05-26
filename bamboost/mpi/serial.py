"""
Mock module for `mpi4py.MPI` to be used when MPI is not available or usage not desired.
Not importing MPI increases launch speed significantly, which is important for CLI
applications.

Used as the default communicator when config.options.mpi is False.
Used when MPI is not available or not needed (i.e. not running under an MPI launcher).

Attributes:
    SerialComm: Minimal serial communicator class (rank=0, size=1).
    NullComm: Placeholder for COMM_NULL that raises on any attribute access.
    COMM_WORLD: Global serial communicator instance.
    COMM_SELF: Per-process serial communicator instance (distinct from COMM_WORLD).
    COMM_NULL: Null communicator sentinel — raises on use.
"""


class SerialComm:
    rank: int = 0
    size: int = 1

    def barrier(self) -> None:
        """No-op in serial mode."""

    def bcast(self, data, root: int = 0):
        """Identity broadcast — returns *data* unchanged."""
        return data

    def scatter(self, data, root: int = 0):
        """Identity scatter — returns *data* unchanged."""
        return data

    def gather(self, data, root: int = 0):
        """Identity gather — returns *data* unchanged."""
        return data

    def allgather(self, data):
        """Return a single-element list containing *data* (serial all-gather)."""
        return [data]

    def allreduce(self, data, op=None):
        """Identity all-reduce — returns *data* unchanged."""
        return data

    def reduce(self, data, op=None, root: int = 0):
        """Identity reduce — returns *data* unchanged."""
        return data


class NullComm:
    """Sentinel for `COMM_NULL` — raises `RuntimeError` on any attribute access.

    In real MPI, `COMM_NULL` is an invalid communicator handle; attempting to use
    it is a programming error.  This class makes such errors visible immediately.
    """

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"COMM_NULL: attempted to use a null communicator (attribute: {name!r}). "
            "This is likely a programming error."
        )


# operators
SUM = lambda a, b: a + b


COMM_WORLD: SerialComm = SerialComm()
"""Global serial communicator (equivalent of MPI_COMM_WORLD in serial mode)."""

COMM_SELF: SerialComm = SerialComm()
"""Per-process serial communicator (equivalent of MPI_COMM_SELF in serial mode).

Deliberately a *separate* instance from COMM_WORLD so that code paths which
distinguish between the two communicators (e.g. `RootProcessMeta.comm_self`)
can behave correctly in serial tests.
"""

COMM_NULL: NullComm = NullComm()
"""Null communicator sentinel — raises on use."""
