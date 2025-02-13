from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Type, TypedDict, TypeVar, Union

from typing_extensions import ParamSpec, TypeAlias

StrPath: TypeAlias = Union[str, Path]
_T = TypeVar("_T")
_U = TypeVar("_U")
_P = ParamSpec("_P")

SimulationMetadataT = TypedDict(
    "SimulationMetadataT",
    {
        "created_at": datetime,
        "modified_at": datetime,
        "description": str,
        "status": str,
    },
    total=False,
)
SimulationParameterT: TypeAlias = Dict[str, Any]


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
