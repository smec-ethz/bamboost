from functools import reduce
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

from bamboost._typing import _P, _T, _U, StrPath


class PathSet(set[Path]):
    def __init__(self, iterable: Optional[Iterable[StrPath]] = None) -> None:
        super().__init__(Path(arg).resolve().expanduser() for arg in iterable or [])

    def add(self, element: Union[str, Path], /) -> None:
        return super().add(Path(element).resolve().expanduser())


# NOT USED
def maybe_apply(func: Callable[[_T], _U]) -> Callable[[Optional[_T]], Optional[_U]]:
    def function(x: Optional[_T]) -> Optional[_U]:
        if x is not None:
            return func(x)

    return function


_R = TypeVar("_R")


# NOT USED
def compose_while_not_none(
    first_func: Callable[_P, Any], *funcs: Callable
) -> Callable[_P, Optional[Any]]:
    """Compose multiple functions into a single function. The output of each
    function is passed as the input to the next function. The functions are
    applied from left to right. If a function returns `None`, the next function
    is not called and `None` is returned.

    Args:
        *funcs: Functions to compose.

    Returns:
        Callable: The composed function.
    """
    funcs = (first_func, *funcs)

    def function(f: Callable, g: Callable) -> Callable:
        def composed(*x: Any) -> Any:
            result = f(*x)
            if result is None:
                return None
            else:
                return g(result)

        return composed

    return reduce(function, funcs)


def full_class_name(_cls):
    """Returns the full name of a class, including the module name."""
    module = _cls.__module__
    if module is None or module == str.__class__.__module__:
        return _cls.__qualname__
    return ".".join([module, _cls.__qualname__])
