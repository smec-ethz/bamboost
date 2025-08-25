from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, TypedDict, TypeVar, Union

from typing_extensions import ParamSpec, TypeAlias

StrPath: TypeAlias = Union[str, Path]
_T = TypeVar("_T")
_U = TypeVar("_U")
_P = ParamSpec("_P")

# Key, value types
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class SimulationMetadataT(TypedDict, total=False):
    created_at: datetime
    modified_at: datetime
    description: str
    status: str


SimulationParameterT: TypeAlias = Mapping[str, Any]


class _MutabilitySentinel(type):
    """A metaclass for creating mutability sentinel types.

    This metaclass is used to create special types that represent mutability
    states (Mutable and Immutable). It provides custom boolean evaluation and
    string representation for the created types.
    """

    def __new__(cls, name, bases, attrs):
        """Create a new class using this metaclass.

        Args:
            name (str): The name of the class being created.
            bases (tuple): The base classes of the class being created.
            attrs (dict): The attributes of the class being created.

        Returns:
            type: The newly created class.
        """
        return super().__new__(cls, name, bases, attrs)

    def __bool__(self):
        """Determine the boolean value of the class.

        Returns:
            bool: True if the class is Mutable, False otherwise.
        """
        return self is Mutable

    def __repr__(self):
        """
        Get the string representation of the class.

        Returns:
            str: The name of the class.
        """
        return self.__name__


class _Mutability(metaclass=_MutabilitySentinel):
    pass


Mutable = type("Mutable", (_Mutability,), {})
Immutable = type("Immutable", (_Mutability,), {})

_MT = TypeVar("_MT", bound=_Mutability)


# Numpy array protocol
class ArrayLike(Protocol):
    """Protocol for objects that can be treated as array-like structures."""

    def __array__(self) -> Any: ...
    def __len__(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
