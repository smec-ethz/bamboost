"""
This module handles the detection and selection of the appropriate MPI implementation for
Bamboost, either using `mpi4py` for real MPI environments or falling back to a mock MPI
implementation for non-MPI environments.

The detection logic considers user configuration options, environment variables, and the
presence of common MPI-related environment variables.

If `mpi4py` is unavailable in an MPI environment, a fallback to the mock implementation is
also provided.

Usage:
    Instead of importing `mpi4py.MPI` directly, import `bamboost.mpi.MPI` to use the
    appropriate MPI module based on the current environment.

    >>> from bamboost.mpi import MPI

Attributes:
    MPI_ON (bool): Flag indicating whether MPI is detected and enabled.
    MPI (module): The selected MPI module (`mpi4py.MPI` or `bamboost.mpi.serial`).
    log (Logger): Logger instance for this module.

Type Aliases:
    Comm: Union of real and serial MPI communicators, available under TYPE_CHECKING.
"""

from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING, Any, Union

from typing_extensions import TypeAlias

from bamboost._config import config
from bamboost._logger import BAMBOOST_LOGGER

if TYPE_CHECKING:
    from mpi4py.MPI import Comm as _MPIComm  # ty: ignore[unresolved-import]

    from bamboost.mpi.serial import SerialComm as _MockComm

    Comm: TypeAlias = Union[_MPIComm, _MockComm]


log = BAMBOOST_LOGGER.getChild(__name__.split(".")[-1])


def _detect_if_mpi_needed() -> bool:
    # 1. environment variable override (highest priority)
    env_mpi = os.environ.get("BAMBOOST_MPI", None)
    if env_mpi is not None:
        if env_mpi in ("0", "false", "False"):
            return False
        if env_mpi in ("1", "true", "True"):
            return True

    # 2. config opt-in for mpi
    if config.options.mpi:
        # Check if any of the common MPI environment variables are set
        # fmt: off
        mpi_env_vars = {
            "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",        # Open MPI
            "PMI_SIZE", "PMI_RANK",                                # MPICH and Intel MPI
            "MV2_COMM_WORLD_SIZE", "MV2_COMM_WORLD_RANK",          # MVAPICH
            "I_MPI_RANK", "I_MPI_SIZE",                            # Intel MPI
            "SLURM_PROCID", "SLURM_NTASKS",                        # SLURM
            "MPI_LOCALNRANKS", "MPI_LOCALRANKID"                   # General/Other
        }
        # fmt: on
        if mpi_env_vars.intersection(os.environ):
            return True

        log.warning(
            "MPI is enabled in the config, but no standard MPI launcher environment "
            "variables were detected. This may indicate that the script is not being "
            "run with MPI. Falling back to mock MPI."
        )

    return False


def _assert_h5py_has_mpi_support() -> None:
    import h5py

    if not h5py.get_config().mpi:
        raise RuntimeError(
            "h5py was not built with MPI support, but MPI is required/enabled in bamboost."
            "Set `config.options.mpi = False` to disable MPI support in bamboost."
        )


class _MPIProxy:
    """Proxy class to delay the import of the MPI module until it's actually needed."""

    _mpi_module: Any = None
    enabled: bool = False

    @classmethod
    def set_from_ctx(cls):
        _should_use_mpi = _detect_if_mpi_needed()
        if _should_use_mpi:
            try:
                _assert_h5py_has_mpi_support()
                from mpi4py import MPI  # ty: ignore[unresolved-import]

                cls._mpi_module = MPI
                cls.enabled = True
            except ImportError:
                log.error(
                    "MPI is required/enabled but `mpi4py` is not installed. "
                    "To bypass this error, either install `mpi4py` or set `config.options.mpi = False` to disable MPI support."
                )
                raise
        else:
            from bamboost.mpi import serial

            cls._mpi_module = serial
            cls.enabled = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mpi_module, name)


_MPIProxy.set_from_ctx()
MPI = _MPIProxy()


class ReuseComm:
    """Marker class to indicate that a communicator should be reused from a parent object.

    This is used in the `Communicator` descriptor to allow child objects to automatically
    reuse the same communicator as their parent without needing to explicitly pass it around.
    """

    def __init__(self, parent_obj: Any):
        self.parent_obj = parent_obj


class _WeakKeyDict:
    """A dictionary that stores keys as weak references, but supports unhashable keys
    by using the key's object identity `id()` for underlying storage.
    """

    def __init__(self) -> None:
        self._data: dict[int, tuple[weakref.ReferenceType, Any]] = {}

    def __setitem__(self, key: Any, value: Any) -> None:
        obj_id = id(key)

        def cleanup(ref_obj: Any) -> None:
            self._data.pop(obj_id, None)

        self._data[obj_id] = (weakref.ref(key, cleanup), value)

    def __getitem__(self, key: Any) -> Any:
        obj_id = id(key)
        if obj_id in self._data:
            ref_obj, value = self._data[obj_id]
            if ref_obj() is not None:
                return value
        raise KeyError(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: Any, default: Any = None) -> Any:
        obj_id = id(key)
        if obj_id in self._data:
            return self._data.pop(obj_id)[1]
        return default

    def __contains__(self, key: Any) -> bool:
        return id(key) in self._data


class Communicator:
    _default_comm: Comm = MPI.COMM_WORLD

    # To allow composed objects to reuse the same communicator, we define a lookup table
    # (weakref dict?) that optionally maps instances to other instances to reuse the same
    # comm object. This allows users to set a communicator on a parent object and have
    # child objects automatically use the same communicator without needing to explicitly
    # pass it around.
    _child_to_parent_map: _WeakKeyDict = _WeakKeyDict()
    _instance_comms: _WeakKeyDict = _WeakKeyDict()

    def __set__(self, instance, value) -> None:
        if instance is None:
            raise AttributeError("Communicator cannot be set on the class.")

        # if the value is a tuple, we assume the second element is the parent object to
        # reuse the comm from
        if isinstance(value, ReuseComm):
            # register the parent-child relationship
            self._child_to_parent_map[instance] = value.parent_obj
        else:
            self._instance_comms[instance] = value
            # clear any old parent relationship if it exists
            self._child_to_parent_map.pop(instance, None)

    def __get__(self, instance, owner) -> Comm:
        if instance is None:
            return self._default_comm

        # if the instance doesn't have an explicitly set comm, we check if it has a parent
        # to reuse we need to traverse the parent chain until we find an explicitly set
        # comm or run out of parents
        current = instance
        visited = set()
        while current is not None:
            # the visited set is to prevent infinite loops in case of circular
            # parent-child relationships. this is unlikely to happen in practice
            curr_id = id(current)
            if curr_id in visited:
                break
            visited.add(curr_id)

            if current in self._instance_comms:
                return self._instance_comms[current]

            current = self._child_to_parent_map.get(current)

        # if we reach here, it means no parent in the chain has an explicitly set comm, so
        # we return the default comm
        return self._default_comm

    def __delete__(self, instance):
        self._instance_comms.pop(instance, None)
        self._child_to_parent_map.pop(instance, None)
