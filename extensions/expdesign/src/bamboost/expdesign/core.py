"""
This module provides utilities to define parameter objects with nested classes,
and a class for defining a experimental design for such parameter configurations.
"""

import math
import copy
import inspect
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    Callable,
    Iterator,
    Iterable,
    overload,
    dataclass_transform,
    get_type_hints,
    cast,
)

from bamboost.core.utilities import flatten_dict, unflatten_dict
from bamboost.parser import _parse_value


KEY_SEPERATOR = "."


class AttrPath:
    def __init__(self, path: tuple):
        self.path = path

    def __getattr__(self, name):
        # Allow chaining: e.g., MyParams.mesh.size -> ('mesh', 'size')
        return AttrPath(self.path + (name,))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.resolve(instance)

    def resolve(self, instance):
        val = instance
        for attr in self.path:
            val = getattr(val, attr)
        return val

    def to_key(self) -> str:
        return KEY_SEPERATOR.join(self.path)


class _ForwardAttributes(type):
    def __getattr__(cls, key):
        if hasattr(cls, "__annotations__") and key in cls.__annotations__:
            return AttrPath((key,))
        # Otherwise, fall back to standard Python behavior
        raise AttributeError(f"type object '{cls.__name__}' has no attribute '{key}'")


@dataclass_transform(kw_only_default=True)
class ParamGraph[RT: ParamGraph](metaclass=_ForwardAttributes):
    """
    Base class for interconnected configuration parameters.

    Provides a tree-like interface for intuitive organization while supporting
    a dependency graph of calculated values, ensuring consistent state 
    propagation across the configuration tree.

    It is parameterized by RT, which defines the root configuration type,
    ensuring that the :attr:`root` property returns the correct type.
    """

    _parent: Any | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 1. Define the __init__ function dynamically
        def __init__(self, **kwargs) -> None:
            for name in cls.__annotations__:
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                    attr = getattr(self, name)
                    if isinstance(attr, ParamGraph):
                        setattr(attr, "_parent", self)
                else:
                    raise TypeError(f"Missing required argument: {name}")

        # 2. Attach it to the class
        setattr(cls, "__init__", __init__)

    @property
    def root(self) -> RT:
        """Returns the root instance of the configuration tree."""
        if self._parent:
            return self._parent.root
        else:
            return cast(RT, self)

    @property
    def dependent_params(self) -> dict[str, Any]:
        """Returns a recursive dictionary of all members marked as @dependent_param."""
        results = {}

        descriptors = inspect.getmembers(
            type(self), predicate=lambda m: isinstance(m, dependent_param)
        )

        for name, _ in descriptors:
            results[name] = getattr(self, name)

        for name, value in self.__dict__.items():
            if name == "_parent":
                continue

            # Only recurse if it's a child Parameters instance
            if isinstance(value, ParamGraph) and value is not self:
                child_results = value.dependent_params

                if child_results:
                    results[name] = child_results

        return results

    @property
    def base_params(self) -> dict[str, Any]:
        """Returns a dictionary representation of the parameters."""
        return self.asdict()

    def asdict(self) -> dict[str, Any]:
        """Recursively converts the parameters into a dictionary."""
        data_dict = {}
        for name, value in self.__dict__.items():
            if name == "_parent":
                continue
            if isinstance(value, ParamGraph):
                data_dict[name] = value.asdict()
            else:
                data_dict[name] = copy.deepcopy(value)
        return data_dict

    @classmethod
    def from_dict(cls, data_dict) -> Self:
        """Creates a new instance from a dictionary of parameters."""
        type_hints = get_type_hints(cls)
        init_kwargs = {}  # may contain instantiated ParamClasses (if nested)

        for name, value in data_dict.items():
            target_type = type_hints.get(name)
            # If the value is a dict, and the target type is a ParamClass --> RECURSION :O
            if (
                isinstance(value, dict)
                and isinstance(target_type, type)
                and issubclass(target_type, ParamGraph)
            ):
                init_kwargs[name] = target_type.from_dict(value)
            else:
                init_kwargs[name] = _parse_value(
                    field_type=target_type,
                    val=value,
                    field_name=name,
                    class_name=cls.__name__,
                )

        return cls(**init_kwargs)

    @classmethod
    def from_toml(cls, path: Path, key: str | None = None) -> Self:
        import tomllib
        
        with open(path, "rb") as f:
            data = tomllib.load(f)
        
        if key:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in TOML file at {path}")
            data = data[key]
            
        return cls.from_dict(data)


class dependent_param[RT](property):
    """
    A property subclass used as a decorator mark properties as dependent parameters.
    """

    if TYPE_CHECKING:

        def __init__(self, fn: Callable[..., RT]): ...

        @overload
        def __get__(self, instance: None, owner: type, /) -> Self:
            """Return an attribute of instance, which is of type owner."""

        @overload
        def __get__(self, instance: Any, owner: type | None = None, /) -> RT: ... # type: ignore


class ExperimentalDesign[TParams: ParamGraph]:
    """Manages the creation of an experimental setup via parameter variations.

    This class takes a base configuration and a set of parameter paths to vary.
    When iterated, it generates the Cartesian product of all specified
    variations, returning new instances of the base parameter object with
    each unique combination applied.

    Example:
        my_params = MyParams(...)
        design = ExperimentalDesign(my_params)
        # accessing an attribute on the class will return a AttrPath object
        design.add_variation(MyParams.mesh.size, [0.01, 0.02, 0.03])
        
        for combo in design:
            run_simulation(combo)

    Attributes:
        base: The baseline parameters object used as a template.
        variations: A collection of parameter paths and their corresponding set of values to be varied.
    """

    def __init__(self, base: TParams):
        self.base: TParams = base
        self.variations: set[tuple[AttrPath, tuple[Any, ...]]] = set()

    def add_variation(self, param: Any, values: Iterable[Any]) -> None:
        """Register a parameter path and its values for the design.

        Args:
            param: The path to the attribute within the configuration.
            values: A sequence of values to iterate over.
        """
        assert isinstance(param, AttrPath)
        self.variations.add((param, tuple(values)))

    def __iter__(self) -> Iterator[TParams]:
        from itertools import product

        keys = [p.to_key() for p, _ in self.variations]
        values = [v for _, v in self.variations]

        base_cls = type(self.base)
        base_dict = self.base.asdict()

        for combination in product(*values):
            variation_dict = {k: v for k, v in zip(keys, combination)}
            base_dict_copy = copy.deepcopy(base_dict)
            new_dict = flatten_dict(base_dict_copy) | variation_dict

            yield base_cls.from_dict(unflatten_dict(new_dict))

    def map[TOut](self, func: Callable[[TParams], TOut]) -> Iterable[TOut]:
        """Returns a lazy iterable yielding transformed parameter sets."""
        return (func(item) for item in self)

    def __len__(self) -> int:
        return math.prod(len(values) for _, values in self.variations)
