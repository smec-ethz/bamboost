"""
Serial communicator backend for bamboost's MPI layer.

Used when MPI is not available or not needed (i.e. not running under an MPI launcher).
Provides a first-class serial implementation rather than a broad MPI mock, so that
operations which are only meaningful in a multi-rank context raise ``NotImplementedError``
instead of silently returning wrong values.

Attributes:
    SerialComm: Minimal serial communicator class (rank=0, size=1).
    NullComm: Placeholder for COMM_NULL that raises on any attribute access.
    COMM_WORLD: Global serial communicator instance.
    COMM_SELF: Per-process serial communicator instance (distinct from COMM_WORLD).
    COMM_NULL: Null communicator sentinel — raises on use.
"""

from __future__ import annotations


class SerialComm:
    """Minimal serial communicator for use when MPI is not available.

    Supports only the operations that are meaningful in a single-process context.
    Any operation that requires multiple ranks (``scatter``, ``gather``, ``send``,
    ``recv``, ``allreduce``, etc.) raises :exc:`NotImplementedError` with a clear
    message directing the user to install ``mpi4py``.
    """

    rank: int = 0
    size: int = 1

    def barrier(self) -> None:
        """No-op in serial mode."""

    def bcast(self, data, root: int = 0):
        """Identity broadcast — returns *data* unchanged."""
        return data

    def scatter(self, data, root: int = 0):
        raise NotImplementedError(
            "scatter() requires MPI (mpi4py). "
            "Install mpi4py or do not call scatter() in serial mode."
        )

    def gather(self, data, root: int = 0):
        raise NotImplementedError(
            "gather() requires MPI (mpi4py). "
            "Install mpi4py or do not call gather() in serial mode."
        )

    def allgather(self, data):
        """Return a single-element list containing *data* (serial all-gather)."""
        return [data]

    def allreduce(self, data, op=None):
        raise NotImplementedError(
            "allreduce() requires MPI (mpi4py). "
            "Install mpi4py or do not call allreduce() in serial mode."
        )

    def reduce(self, data, op=None, root: int = 0):
        raise NotImplementedError(
            "reduce() requires MPI (mpi4py). "
            "Install mpi4py or do not call reduce() in serial mode."
        )

    def send(self, data, dest: int, tag: int = 0) -> None:
        raise NotImplementedError(
            "send() requires MPI (mpi4py). "
            "Install mpi4py or do not call send() in serial mode."
        )

    def recv(self, source: int = 0, tag: int = 0):
        raise NotImplementedError(
            "recv() requires MPI (mpi4py). "
            "Install mpi4py or do not call recv() in serial mode."
        )


class NullComm:
    """Sentinel for ``COMM_NULL`` — raises :exc:`RuntimeError` on any attribute access.

    In real MPI, ``COMM_NULL`` is an invalid communicator handle; attempting to use
    it is a programming error.  This class makes such errors visible immediately.
    """

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"COMM_NULL: attempted to use a null communicator (attribute: {name!r}). "
            "This is likely a programming error."
        )


COMM_WORLD: SerialComm = SerialComm()
"""Global serial communicator (equivalent of MPI_COMM_WORLD in serial mode)."""

COMM_SELF: SerialComm = SerialComm()
"""Per-process serial communicator (equivalent of MPI_COMM_SELF in serial mode).

Deliberately a *separate* instance from COMM_WORLD so that code paths which
distinguish between the two communicators (e.g. ``RootProcessMeta.comm_self``)
behave correctly in serial tests.
"""

COMM_NULL: NullComm = NullComm()
"""Null communicator sentinel — raises on use."""
