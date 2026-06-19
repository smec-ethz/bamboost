import dataclasses
import typing
from enum import Enum
from collections.abc import Mapping
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import numpy as np

T = TypeVar("T")


def _to_list(val: Any) -> Any:
    """Normalizes sequence inputs (including NumPy arrays) to standard Python lists."""
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def _parse_union(args: tuple, val: Any, field_name: str, class_name: str) -> Any:
    """Handles Union and Optional types by attempting to parse each union member."""
    last_err = None
    for arg in args:
        if arg is type(None):
            continue
        try:
            return _parse_value(arg, val, field_name, class_name)
        except (TypeError, ValueError) as e:
            last_err = e
            continue

    if type(None) in args and val is None:
        return None

    raise last_err or TypeError(
        f"Value '{val}' for field '{field_name}' in {class_name} does not match any type in {args}"
    )


def _parse_tuple(args: tuple, val: Any, field_name: str, class_name: str) -> tuple:
    """Parses and validates tuple fields, supporting fixed-length, variadic, and unannotated tuples."""
    val = _to_list(val)
    if not isinstance(val, (list, tuple)):
        raise TypeError(
            f"Expected sequence for field '{field_name}' in {class_name}, got {type(val).__name__}"
        )

    # tuple[T, ...] (variadic)
    if len(args) == 2 and args[1] is Ellipsis:
        item_type = args[0]
        return tuple(_parse_value(item_type, x, field_name, class_name) for x in val)

    # tuple[T1, T2, ...] (fixed length)
    elif len(args) > 0:
        if len(val) != len(args):
            raise ValueError(
                f"Expected tuple of length {len(args)} for field '{field_name}' in {class_name}, got length {len(val)}"
            )
        return tuple(
            _parse_value(t, x, field_name, class_name) for t, x in zip(args, val)
        )

    # tuple (unannotated)
    return tuple(val)


def _parse_list(args: tuple, val: Any, field_name: str, class_name: str) -> list:
    """Parses and validates list fields."""
    val = _to_list(val)
    if not isinstance(val, (list, tuple)):
        raise TypeError(
            f"Expected sequence for field '{field_name}' in {class_name}, got {type(val).__name__}"
        )
    item_type = args[0] if args else Any
    return [_parse_value(item_type, x, field_name, class_name) for x in val]


def _parse_set(args: tuple, val: Any, field_name: str, class_name: str) -> set:
    """Parses and validates set fields."""
    val = _to_list(val)
    if not isinstance(val, (list, tuple, set)):
        raise TypeError(
            f"Expected sequence for field '{field_name}' in {class_name}, got {type(val).__name__}"
        )
    item_type = args[0] if args else Any
    return {_parse_value(item_type, x, field_name, class_name) for x in val}


def _parse_mapping(args: tuple, val: Any, field_name: str, class_name: str) -> Mapping:
    """Parses and validates generic mapping / dictionary fields."""
    if not isinstance(val, Mapping):
        raise TypeError(
            f"Expected Mapping for field '{field_name}' in {class_name}, got {type(val).__name__}"
        )
    key_type = args[0] if args else Any
    val_type = args[1] if args else Any
    return {
        _parse_value(key_type, k, field_name, class_name): _parse_value(
            val_type, v, field_name, class_name
        )
        for k, v in val.items()
    }


def _parse_value(
    field_type: Any,
    val: Any,
    field_name: str,
    class_name: str,
) -> Any:
    """Recursively parses and validates a value against the specified type hint."""
    # 1. Handle Any
    if field_type is Any:
        return val

    # Extract generic components if present
    origin = get_origin(field_type) or field_type
    args = get_args(field_type)

    # 2. Handle Union / Optional types
    if origin is typing.Union or getattr(origin, "__name__", None) == "UnionType":
        return _parse_union(args, val, field_name, class_name)

    # 3. Handle Literal types
    if origin is typing.Literal:
        if val in args:
            return val
        raise ValueError(
            f"Value '{val}' for field '{field_name}' in {class_name} is not one of {args}"
        )

    # 4. Handle None / Optional
    if val is None:
        if field_type is type(None):
            return None
        raise TypeError(
            f"Field '{field_name}' in {class_name} cannot be None (expected {field_type})"
        )

    # 5. Handle nested dataclasses
    if dataclasses.is_dataclass(origin):
        return dataclass_from_dict(field_type, val)

    # Unify generic and non-generic classes
    concrete_type = origin if origin is not None else field_type

    if not isinstance(concrete_type, type):
        # Fallback for complex typing constructs that are not classes
        return val

    # 6. Handle Enums
    if issubclass(origin, Enum):
        return origin(val)
    # 7. Parse collections, arrays, and primitive types
    if issubclass(concrete_type, tuple):
        return _parse_tuple(args, val, field_name, class_name)

    if issubclass(concrete_type, list):
        return _parse_list(args, val, field_name, class_name)

    if issubclass(concrete_type, set):
        return _parse_set(args, val, field_name, class_name)

    if issubclass(concrete_type, Mapping):
        return _parse_mapping(args, val, field_name, class_name)

    if issubclass(concrete_type, np.ndarray):
        return np.asarray(val)

    if issubclass(concrete_type, float) and isinstance(val, (int, float)):
        return float(val)

    if (
        issubclass(concrete_type, int)
        and isinstance(val, (int, float))
        and not isinstance(val, bool)
    ):
        return int(val)

    if issubclass(concrete_type, str):
        return str(val)

    if issubclass(concrete_type, bool):
        return bool(val)

    # Fallback to direct nominal type constructor or type check
    if isinstance(val, concrete_type):
        return val
    try:
        return concrete_type(val)
    except Exception as e:
        raise TypeError(
            f"Could not convert value {val} to type {field_type} for field '{field_name}' in {class_name}"
        ) from e


def dataclass_from_dict(cls: type[T], data: Any) -> T:
    """Instantiate a dataclass from a dictionary recursively, supporting type conversions,
    nested dataclasses, and standard collections. Fully handles numpy arrays / lists to tuples.
    """
    if not dataclasses.is_dataclass(cls):
        return data

    if not isinstance(data, Mapping):
        raise TypeError(
            f"Expected a Mapping for dataclass {cls.__name__}, got {type(data).__name__}"
        )

    type_hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    init_kwargs = {}

    for field_name, field in fields.items():
        if not field.init:
            continue

        field_type = type_hints[field_name]

        if field_name not in data:
            if field.default is not dataclasses.MISSING:
                init_kwargs[field_name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                init_kwargs[field_name] = field.default_factory()
            else:
                raise ValueError(
                    f"Missing required field '{field_name}' for dataclass {cls.__name__}"
                )
            continue

        val = data[field_name]
        init_kwargs[field_name] = _parse_value(
            field_type, val, field_name, cls.__name__
        )

    return cls(**init_kwargs)
