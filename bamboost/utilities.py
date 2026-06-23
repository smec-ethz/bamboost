from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from bamboost._typing import StrPath


class PathSet(set[Path]):
    def __init__(self, iterable: Optional[Iterable[StrPath]] = None) -> None:
        super().__init__(Path(arg).expanduser().resolve() for arg in iterable or [])

    def add(self, element: Union[str, Path], /) -> None:
        return super().add(Path(element).expanduser().resolve())


class ComparableIterable:
    def __init__(self, ori):
        self.ori = np.asarray(ori)

    def __eq__(self, other):
        other = np.asarray(other)
        if other.shape != self.ori.shape:
            return False
        return (other == self.ori).all()


def full_class_name(_cls):
    """Returns the full name of a class, including the module name."""
    module = _cls.__module__
    if module is None or module == str.__class__.__module__:
        return _cls.__qualname__
    return f"{module}.{_cls.__qualname__}"
