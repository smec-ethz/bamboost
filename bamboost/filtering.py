# ty: ignore[invalid-method-override]
from __future__ import annotations

import json
import operator
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, Union, overload

from bamboost.utilities import ComparableIterable

# Type for operands in expressions. These are dtypes compatible with pandas DataFrame
# columns. (see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html)
Operand = Union["Key", "Operator", str, float, int, datetime, timedelta]
Numeric = float | int


def _json_encoder(obj: Any) -> Any:
    if isinstance(obj, (datetime, timedelta)):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class _SupportsOperators:
    if TYPE_CHECKING:

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
        def __floordiv__(self, other: Operand) -> "Operator": ...
        def __mod__(self, other: Operand) -> "Operator": ...
        def __pow__(self, other: Operand) -> "Operator": ...
        def __or__(self, other: Operand) -> "Or": ...
        def __and__(self, other: Operand) -> "And": ...

    def isin(self, values: Iterable[Any]) -> "Operator":
        return Operator("in", self, values)

    def contains(self, substring: str) -> "Operator":
        return Operator("contains", self, substring)


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

    def make_and(self, other):
        return And(self, other)

    def make_or(self, other):
        return Or(self, other)

    cls.__and__ = make_and
    cls.__or__ = make_or

    return cls


@add_operators
class Operator(_SupportsOperators):
    @overload
    def __init__(
        self,
        op: Callable[[Any, Any], bool] | str,
        a: Numeric | str | Key | _SupportsOperators,
        b: Numeric | str | Key | _SupportsOperators | Iterable,
    ) -> None: ...
    @overload
    def __init__(
        self, op: Callable[[Any], bool] | str, a: Numeric | str | Key | Operator
    ) -> None: ...
    def __init__(self, op, a, b=None):
        self._op = op
        self._a = a
        self._b = b

    def evaluate(self, item: Any) -> Any:
        def resolve(val):
            if isinstance(val, Key):
                return item[val._value]
            elif isinstance(val, (Operator, And, Or)):
                return val.evaluate(item)
            return val

        if self._op == "in":
            return resolve(self._a).isin(self._b)
        if self._op == "contains":
            return resolve(self._a).str.contains(self._b)

        if self._b is None:
            return self._op(resolve(self._a))

        return self._op(resolve(self._a), resolve(self._b))

    def to_dict(self) -> dict[str, Any]:
        op_map = {
            operator.lt: "Lt",
            operator.le: "Lte",
            operator.eq: "Eq",
            operator.ne: "Ne",
            operator.gt: "Gt",
            operator.ge: "Gte",
        }

        def resolve(val):
            if isinstance(val, Key):
                return {"type": "Key", "name": val._value}
            elif isinstance(val, (Operator, And, Or)):
                return val.to_dict()
            return {"type": "Value", "value": val}

        if self._op == "in":
            return {
                "type": "In",
                "left": resolve(self._a),
                "right": resolve(
                    self._b.ori if isinstance(self._b, ComparableIterable) else self._b
                ),
            }
        if self._op == "contains":
            return {"type": "Contains", "left": resolve(self._a), "substring": self._b}

        return {
            "type": "Compare",
            "op": op_map.get(self._op, str(self._op)),
            "left": resolve(self._a),
            "right": resolve(self._b),
        }

    def __repr__(self) -> str:
        op_name = self._op if isinstance(self._op, str) else self._op.__name__
        return f"Operation({self._a} {op_name} {self._b})"


@add_operators
class And(_SupportsOperators):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, item):
        return self.left.evaluate(item) & self.right.evaluate(item)

    def to_dict(self):
        return {
            "type": "And",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@add_operators
class Or(_SupportsOperators):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, item):
        return self.left.evaluate(item) | self.right.evaluate(item)

    def to_dict(self):
        return {
            "type": "Or",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@add_operators
class Key(_SupportsOperators):
    def __init__(self, key: str) -> None:
        self._value = key

    def __repr__(self) -> str:
        return f"Key({self._value})"


class Filter:
    """Filter applied to a collection."""

    def __init__(
        self, *operators: Operator | And | Or, tags: Iterable[str] | None = None
    ) -> None:
        self._ops = operators
        self._tags: set[str] = set(tags) if tags else set()

    def to_dict(self) -> dict[str, Any] | None:
        # first: add tags to the filter if they exist
        if self._tags:
            tag_filter = Operator("in", Key("tags"), self._tags)
            self._ops = (tag_filter, *self._ops)

        if not self._ops:
            return None
        combined = self._ops[0]
        for op in self._ops[1:]:
            combined = combined & op
        return combined.to_dict()

    def to_string(self) -> str:
        return json.dumps(self.to_dict(), default=_json_encoder)

    def __and__(self, other: Filter | None) -> Filter:
        return (
            Filter(*self._ops, *other._ops, tags=self._tags.union(other._tags))
            if other
            else self
        )

    def __repr__(self) -> str:
        return "Filter({})".format(" & ".join(str(op) for op in self._ops))


class SortInstruction:
    def __init__(self, key: str, ascending: bool = True) -> None:
        self.key = key
        self.ascending = ascending

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "ascending": self.ascending}

    def __repr__(self) -> str:
        order = "ASC" if self.ascending else "DESC"
        return f"SortInstruction({self.key} {order})"


class Sorter:
    """Sorter applied to a collection."""

    def __init__(self, *instructions: SortInstruction) -> None:
        self._instructions: Sequence[SortInstruction] = instructions

    def to_list(self) -> list[dict[str, Any]]:
        return [instr.to_dict() for instr in self._instructions]

    def to_string(self) -> str:
        return json.dumps(self.to_list(), default=_json_encoder)

    def __and__(self, other: Sorter | None) -> Sorter:
        return Sorter(*self._instructions, *other._instructions) if other else self

    def __repr__(self) -> str:
        return "Sorter({})".format(
            ", ".join(str(instr) for instr in self._instructions)
        )
