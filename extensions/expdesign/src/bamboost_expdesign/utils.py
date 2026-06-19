"""
Utilities related to dataclasses.
"""

from typing import Any, Callable
from dataclasses import is_dataclass, fields, asdict


def is_dataclass_instance(obj: Any) -> bool:
    return is_dataclass(obj) and not isinstance(obj, type)


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass_instance(obj):
        return asdict(obj)
    
    raise TypeError(f"Expected a dataclass instance, got {type(obj)}")


def deep_replace[T](obj: T, /, **kwargs) -> T:
    """
    Like dataclasses.replace but can replace an arbitrarily nested attribute.
    https://gist.github.com/mgaitan/94bc1483a3efa399b2f5052ff3827f24
    """
    from dataclasses import replace
    from operator import attrgetter

    assert is_dataclass_instance(obj)

    for k, v in kwargs.items():
        k = k.replace("__", ".")
        
        while "." in k:
            prefix, _, attr = k.rpartition(".")
            deep_attr = attrgetter(prefix)(obj)
            v = replace(deep_attr, **{attr: v})
            k = prefix
        obj = replace(obj, **{k: v})
    return obj


def collect_members_recursive(
    obj: Any, 
    predicate: Callable[[Any], bool], 
) -> dict[str, Any]:
    """
    Finds all members of obj satisfying predicate() and recurses into nested structures.
    """
    import inspect

    if not is_dataclass_instance(obj):
        return {}

    result = {}
    
    # 1. Check class members (properties/methods)
    for name, member in inspect.getmembers(type(obj)):
        if predicate(member):
            result[name] = getattr(obj, name)

    # 2. Recurse into fields
    for f in fields(obj):
        val = getattr(obj, f.name)
        if is_dataclass_instance(val):
            nested = collect_members_recursive(val, predicate)
            if nested:
                result[f.name] = nested
                
    return result