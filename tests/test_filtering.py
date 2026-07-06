from bamboost.filtering import And, Or

"""Tests for bamboost.index._filtering module."""
import operator
import pandas as pd
import pytest
from bamboost.filtering import Filter, Key, Operator, Sorter, SortInstruction


class TestKey:
    """Tests for _Key class."""

    def test_key_initialization(self):
        """Test _Key initialization."""
        key = Key("test_key")
        assert key._value == "test_key"

    def test_key_repr(self):
        """Test _Key string representation."""
        key = Key("my_key")
        assert repr(key) == "Key(my_key)"

    def test_key_equality_operator(self):
        """Test _Key equality operator."""
        key = Key("value")
        op = key == 5
        assert isinstance(op, Operator)

    def test_key_comparison_operators(self):
        """Test _Key comparison operators."""
        key = Key("age")
        op_lt = key < 30
        assert isinstance(op_lt, Operator)
        op_gt = key > 20
        assert isinstance(op_gt, Operator)
        op_le = key <= 25
        assert isinstance(op_le, Operator)
        op_ge = key >= 25
        assert isinstance(op_ge, Operator)
        op_ne = key != 10
        assert isinstance(op_ne, Operator)

    def test_key_arithmetic_operators(self):
        """Test _Key arithmetic operators."""
        key = Key("number")
        op_add = key + 5
        assert isinstance(op_add, Operator)
        op_sub = key - 3
        assert isinstance(op_sub, Operator)
        op_mul = key * 2
        assert isinstance(op_mul, Operator)
        op_div = key / 2
        assert isinstance(op_div, Operator)

    def test_key_logical_operators(self):
        """Test _Key logical operators."""
        key1 = Key("flag1")
        key2 = Key("flag2")
        op_and = key1 & key2
        assert isinstance(op_and, And)
        op_or = key1 | key2
        assert isinstance(op_or, Or)


class TestOperator:
    """Tests for Operator class."""

    def test_operator_initialization_binary(self):
        """Test Operator initialization with binary operation."""
        key = Key("value")
        op = Operator(operator.eq, key, 5)
        assert op._op == operator.eq
        assert op._a == key
        assert op._b == 5

    def test_operator_initialization_unary(self):
        """Test Operator initialization with unary operation."""
        key = Key("value")
        op = Operator(operator.not_, key)
        assert op._op == operator.not_
        assert op._a == key
        assert op._b is None

    def test_operator_evaluate_equality(self):
        """Test Operator evaluation with equality."""
        key = Key("age")
        op = key == 30
        item = {"age": 30}
        assert op.evaluate(item) is True
        item = {"age": 25}
        assert op.evaluate(item) is False

    def test_operator_evaluate_comparison(self):
        """Test Operator evaluation with comparison operators."""
        key = Key("score")
        op_gt = key > 80
        assert op_gt.evaluate({"score": 90}) is True
        assert op_gt.evaluate({"score": 70}) is False
        op_lt = key < 50
        assert op_lt.evaluate({"score": 40}) is True
        assert op_lt.evaluate({"score": 60}) is False

    def test_operator_evaluate_nested(self):
        """Test Operator evaluation with nested operators."""
        key = Key("value")
        op = key + 5 > 10
        assert op.evaluate({"value": 6}) is True
        assert op.evaluate({"value": 5}) is False
        assert op.evaluate({"value": 4}) is False

    def test_operator_repr(self):
        """Test Operator string representation."""
        key = Key("age")
        op = key == 30
        repr_str = repr(op)
        assert "Operation" in repr_str
        assert "eq" in repr_str

    def test_operator_chaining(self):
        """Test chaining multiple operators."""
        key = Key("x")
        op = (key + 5) * 2 > 20
        assert op.evaluate({"x": 6}) is True
        assert op.evaluate({"x": 5}) is False


class TestFilter:
    """Tests for Filter class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
                "score": [85, 90, 75, 95],
                "tags": [["new"], ["old", "special"], ["new"], ["old"]],
            }
        )

    def test_filter_initialization_empty(self):
        """Test Filter initialization with no operators."""
        f = Filter()
        assert len(f._ops) == 0
        assert len(f._tags) == 0

    def test_filter_initialization_with_operators(self):
        """Test Filter initialization with operators."""
        key = Key("age")
        op = key > 30
        f = Filter(op)
        assert len(f._ops) == 1
        assert f._ops[0] == op

    def test_filter_initialization_with_tags(self):
        """Test Filter initialization with tags."""
        f = Filter(tags=["tag1", "tag2"])
        assert f._tags == {"tag1", "tag2"}

    def test_filter_and_operator(self):
        """Test Filter.__and__ method."""
        key = Key("age")
        f1 = Filter(key > 20)
        f2 = Filter(key < 40)
        combined = f1 & f2
        assert len(combined._ops) == 2
        assert isinstance(combined, Filter)

    def test_filter_and_with_none(self):
        """Test Filter.__and__ with None."""
        key = Key("age")
        f = Filter(key > 20)
        result = f & None
        assert result is f

    def test_filter_and_with_tags(self):
        """Test Filter.__and__ combines tags."""
        f1 = Filter(tags=["tag1"])
        f2 = Filter(tags=["tag2"])
        combined = f1 & f2
        assert combined._tags == {"tag1", "tag2"}

    def test_filter_repr(self):
        """Test Filter string representation."""
        key = Key("age")
        f = Filter(key > 20, key < 40)
        repr_str = repr(f)
        assert "Filter" in repr_str


class TestSortInstruction:
    """Tests for SortInstruction class."""

    def test_sort_instruction_initialization_ascending(self):
        """Test SortInstruction initialization with ascending order."""
        instr = SortInstruction("age", ascending=True)
        assert instr.key == "age"
        assert instr.ascending is True

    def test_sort_instruction_initialization_descending(self):
        """Test SortInstruction initialization with descending order."""
        instr = SortInstruction("score", ascending=False)
        assert instr.key == "score"
        assert instr.ascending is False

    def test_sort_instruction_default_ascending(self):
        """Test SortInstruction defaults to ascending."""
        instr = SortInstruction("name")
        assert instr.ascending is True

    def test_sort_instruction_repr(self):
        """Test SortInstruction string representation."""
        instr_asc = SortInstruction("age", ascending=True)
        assert "ASC" in repr(instr_asc)
        assert "age" in repr(instr_asc)
        instr_desc = SortInstruction("age", ascending=False)
        assert "DESC" in repr(instr_desc)


class TestSorter:
    """Tests for Sorter class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "name": ["Charlie", "Alice", "David", "Bob"],
                "age": [35, 25, 40, 30],
                "score": [75, 85, 95, 90],
            }
        )

    def test_sorter_initialization_empty(self):
        """Test Sorter initialization with no instructions."""
        s = Sorter()
        assert len(s._instructions) == 0

    def test_sorter_initialization_with_instructions(self):
        """Test Sorter initialization with sort instructions."""
        instr = SortInstruction("age")
        s = Sorter(instr)
        assert len(s._instructions) == 1
        assert s._instructions[0] == instr

    def test_sorter_and_operator(self):
        """Test Sorter.__and__ method."""
        instr1 = SortInstruction("age")
        instr2 = SortInstruction("score")
        s1 = Sorter(instr1)
        s2 = Sorter(instr2)
        combined = s1 & s2
        assert len(combined._instructions) == 2
        assert isinstance(combined, Sorter)

    def test_sorter_and_with_none(self):
        """Test Sorter.__and__ with None."""
        instr = SortInstruction("age")
        s = Sorter(instr)
        result = s & None
        assert result is s

    def test_sorter_repr(self):
        """Test Sorter string representation."""
        instr1 = SortInstruction("age")
        instr2 = SortInstruction("score", ascending=False)
        s = Sorter(instr1, instr2)
        repr_str = repr(s)
        assert "Sorter" in repr_str
        assert "age" in repr_str
        assert "score" in repr_str


class TestIntegration:
    """Integration tests for Filter and Sorter together."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "age": [25, 30, 35, 40, 28],
                "score": [85, 90, 75, 95, 88],
                "tags": [["new"], ["old"], ["new", "special"], ["old"], ["new"]],
            }
        )
