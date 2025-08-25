from __future__ import annotations

import operator
from datetime import datetime, timedelta
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Sequence, Union, overload

if TYPE_CHECKING:
    from pandas import DataFrame, Series

# Type for operands in expressions. These are dtypes compatible with pandas DataFrame
# columns. (see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html)
Operand = Union["_Key", "Operator", str, float, int, datetime, timedelta]


class _SupportsOperators:
    def __lt__(self, other: Operand) -> "Operator": ...
    def __le__(self, other: Operand) -> "Operator": ...
    def __eq__(self, other: Operand) -> "Operator": ...
    def __ne__(self, other: Operand) -> "Operator": ...
    def __gt__(self, other: Operand) -> "Operator": ...
    def __ge__(self, other: Operand) -> "Operator": ...
    def __add__(self, other: Operand) -> "Operator": ...
    def __sub__(self, other: Operand) -> "Operator": ...
    def __mul__(self, other: Operand) -> "Operator": ...
    def __truediv__(self, other: Operand) -> "Operator": ...


def add_operators(cls):
    ops = {
        "__lt__": operator.lt,
        "__le__": operator.le,
        "__eq__": operator.eq,
        "__ne__": operator.ne,
        "__gt__": operator.gt,
        "__ge__": operator.ge,
        "__add__": operator.add,
        "__sub__": operator.sub,
        "__mul__": operator.mul,
        "__truediv__": operator.truediv,
        "__floordiv__": operator.floordiv,
        "__mod__": operator.mod,
        "__pow__": operator.pow,
    }

    def make_op(op_func):
        def method(self, other):
            return Operator(op_func, self, other)

        return method

    for name, func in ops.items():
        setattr(cls, name, make_op(func))

    return cls


@add_operators
class Operator(_SupportsOperators):
    @overload
    def __init__(
        self,
        op: Callable[[Any, Any], bool],
        a: Number | str | _Key | Operator,
        b: Number | str | _Key | Operator,
    ) -> None: ...
    @overload
    def __init__(
        self, op: Callable[[Any], bool], a: Number | str | _Key | Operator
    ) -> None: ...
    def __init__(self, op, a, b=None):
        self._op = op
        self._a = a
        self._b = b

    def evaluate(self, item: dict) -> bool:
        def resolve(val):
            if isinstance(val, _Key):
                return item[val._value]
            elif isinstance(val, Operator):
                return val.evaluate(item)
            return val

        if self._b is None:
            return self._op(resolve(self._a))
        return self._op(resolve(self._a), resolve(self._b))

    def __repr__(self) -> str:
        return f"Operation({self._a} {self._op.__name__} {self._b})"


@add_operators
class _Key(_SupportsOperators):
    def __init__(self, key: str) -> None:
        self._value = key

    def __repr__(self) -> str:
        return f"Key({self._value})"


class Filter:
    """Filter applied to a collection."""

    def __init__(self, *operators: Operator) -> None:
        self._ops: Sequence[Operator] = operators

    def __and__(self, other: Filter | None) -> Filter:
        return Filter(*self._ops, *other._ops) if other else self

    def apply(self, df: DataFrame) -> DataFrame | Series | Any:
        mask = df.apply(lambda row: all(op.evaluate(row) for op in self._ops), axis=1)
        return df[mask]

    def __repr__(self) -> str:
        return "Filter({})".format(" & ".join(str(op) for op in self._ops))
