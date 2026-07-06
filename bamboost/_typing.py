from pathlib import Path
from typing import Any, Protocol, TypeVar

from typing_extensions import TypeAlias

StrPath: TypeAlias = str | Path
"""Type alias for string or Path-like objects."""


# Marker class for mutable and immutable files
class Mutable:
    pass


class Immutable:
    pass


MT = TypeVar("MT", Mutable, Immutable)


# Numpy array protocol
class ArrayLike(Protocol):
    """Protocol for objects that can be treated as array-like structures."""

    def __array__(self) -> Any: ...
    def __len__(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
