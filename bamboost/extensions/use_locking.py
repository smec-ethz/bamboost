"""This module provides a custom rather primitive file locking mechanism for
HDF5 files.

This module provides a function `use_locking` that installs the mechanism by
Monkey-patching the `FileHandler` class. To use this mechanism, call the
`use_locking` function at the start of your script, before creating any
`bamboost.common.file_handler.FileHandler` instances.

Example:
    >>> from bamboost.extensions.use_locking import use_locking
    >>> use_locking("shared")
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable
from functools import wraps
from io import BufferedReader
from typing import Any, Literal, Tuple

import h5py

from bamboost.common.file_handler import (
    FILE_MODE_HIRARCHY,
    FileHandler,
    log,
    open_h5file,
)


def get_lock_and_open_function(
    lock_type: Literal["shared", "exclusive"],
) -> Callable[..., Tuple[h5py.File, BufferedReader]]:
    """Returns a function that locks and opens an HDF5 file with the specified
    lock type.

    The returned function acquires a lock on the lock file and opens the HDF5
    file, returning both the opened HDF5 file object and the lock file object.

    The returned function takes the following parameters:

    | Argument  | Description                                                                 |
    |-----------|-----------------------------------------------------------------------------|
    | file      | The path to the HDF5 file to open.                                          |
    | lock_file | The path to the lock file.                                                  |
    | mode      | The mode to open the file in (e.g., 'r', 'w', 'a').                         |
    | driver    | The HDF5 driver to use (e.g., 'mpio' for MPI-IO).                           |
    | comm      | The MPI communicator (only used with 'mpio' driver).                        |

    Args:
        lock_type (Literal["shared", "exclusive"]): The type of lock to use.

    Returns:
        Callable: A function that locks and opens an HDF5 file.
    """
    lock_type_int = {"shared": fcntl.LOCK_SH, "exclusive": fcntl.LOCK_EX}[lock_type]

    def lock_and_open(
        file: str, lock_file: str, mode: str, driver: str, comm: Any
    ) -> Tuple[h5py.File, BufferedReader]:
        lock_file_object = open(lock_file, "rb")
        # acquire a lock
        fcntl.flock(lock_file_object, lock_type_int)
        # open the file
        h5_file = open_h5file(file, mode, driver, comm)
        return h5_file, lock_file_object

    return lock_and_open


def close_and_unlock(h5_file: h5py.File, lock_file: BufferedReader) -> None:
    """Close an HDF5 file and release the associated file lock.

    This function performs the following operations:
    1. Closes the given HDF5 file.
    2. Releases the lock on the lock file.
    3. Closes the lock file.

    Args:
        h5_file: The HDF5 file object to be closed.
        lock_file: The lock file object to be unlocked and closed.

    Returns:
        None
    """
    h5_file.close()
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()


def get_open_method(
    lock_type: Literal["shared", "exclusive"],
) -> Callable[..., h5py.File]:
    open_function = get_lock_and_open_function(lock_type)

    @wraps(FileHandler.open)
    def _modified_open(self: FileHandler, mode: str = "r", driver=None, comm=None):
        """Create a primitive lock file for the file.

        Will replace `file_handler.FileHandler.open` method.
        """
        if self._lock <= 0:
            # lock_file_access(self.file_name, comm)

            log.debug(f"[{id(self)}] Open {self.file_name}")
            # self.file_object = open_h5file(self.file_name, mode, driver, comm)
            self.file_object, self._lock_file = open_function(  # type: ignore
                self.file_name, self._lock_file_name, mode, driver, comm
            )

        if FILE_MODE_HIRARCHY[self.file_object.mode] < FILE_MODE_HIRARCHY[mode]:
            close_and_unlock(self.file_object, self._lock_file)
            self.file_object, self._lock_file = open_function(  # type: ignore
                self.file_name, self._lock_file_name, mode, driver, comm
            )

        log.debug(f"[{id(self)}] Lock stack {self._lock}")
        self._lock += 1
        return self.file_object

    return _modified_open


@wraps(FileHandler.close)
def _modified_close(self: FileHandler):
    """Create a primitive lock for the file.

    Will replace `file_handler.FileHandler.close` method.
    """
    self._lock -= 1
    if self._lock == 0:
        log.debug(f"[{id(self)}] Close {self.file_name}")
        # self.file_object.close()
        close_and_unlock(self.file_object, self._lock_file)

    log.debug(f"[{id(self)}] Lock stack {self._lock}")


def _extend_init(original_init):
    @wraps(FileHandler.__init__)
    def _modified_init(self: FileHandler, *args, **kwargs):
        original_init(self, *args, **kwargs)

        self._lock_file_name = f"{self.file_name}.lock"  # type: ignore

        if not os.path.exists(self._lock_file_name):
            with open(self._lock_file_name, "w") as f:
                f.write("LOCK FILE")

    return _modified_init


def use_locking(lock_type: Literal["shared", "exclusive"]) -> None:
    """Installs a primitive locking mechanism for FileHandler operations.

    This function modifies the FileHandler class to include file locking
    capabilities, ensuring thread-safe access to files. It wraps the
    `__init__`, `open`, and `close` methods of FileHandler with locking mechanisms.

    Usage:
        Call this function before using FileHandler to enable locking:
        use_locking("shared")  # or use_locking("exclusive")

    Note:
        This function should be called only once, preferably at the start
        of your program, before any FileHandler instances are created.

    Note:
        You should only need this functionality if you are using HDF5 <= 1.10.6.
        The newer versions of HDF5 (>= 1.10.7) have built-in file locking.

    Args:
        lock_type (Literal["shared", "exclusive"]): The type of lock to use.
            - "shared" allows multiple processes to open the file simultaneously.
            - "exclusive" allows only one reader or writer at a time.
    """
    if not hasattr(FileHandler.__init__, "__wrapped__"):
        FileHandler.__init__ = _extend_init(FileHandler.__init__)

    if not hasattr(FileHandler.open, "__wrapped__"):
        FileHandler.open = get_open_method(lock_type)

    if not hasattr(FileHandler.close, "__wrapped__"):
        FileHandler.close = _modified_close

    log.info("Primitive locking installed")
