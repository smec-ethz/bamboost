"""Tests for bamboost.index._filtering module."""

import operator

import pandas as pd
import pytest

from bamboost.index._filtering import (
    Filter,
    Operator,
    SortInstruction,
    Sorter,
    _Key,
)


class TestKey:
    """Tests for _Key class."""

    def test_key_initialization(self):
        """Test _Key initialization."""
        key = _Key("test_key")
        assert key._value == "test_key"

    def test_key_repr(self):
        """Test _Key string representation."""
        key = _Key("my_key")
        assert repr(key) == "Key(my_key)"

    def test_key_equality_operator(self):
        """Test _Key equality operator."""
        key = _Key("value")
        op = key == 5
        assert isinstance(op, Operator)

    def test_key_comparison_operators(self):
        """Test _Key comparison operators."""
        key = _Key("age")
        
        # Less than
        op_lt = key < 30
        assert isinstance(op_lt, Operator)
        
        # Greater than
        op_gt = key > 20
        assert isinstance(op_gt, Operator)
        
        # Less than or equal
        op_le = key <= 25
        assert isinstance(op_le, Operator)
        
        # Greater than or equal
        op_ge = key >= 25
        assert isinstance(op_ge, Operator)
        
        # Not equal
        op_ne = key != 10
        assert isinstance(op_ne, Operator)

    def test_key_arithmetic_operators(self):
        """Test _Key arithmetic operators."""
        key = _Key("number")
        
        # Addition
        op_add = key + 5
        assert isinstance(op_add, Operator)
        
        # Subtraction
        op_sub = key - 3
        assert isinstance(op_sub, Operator)
        
        # Multiplication
        op_mul = key * 2
        assert isinstance(op_mul, Operator)
        
        # Division
        op_div = key / 2
        assert isinstance(op_div, Operator)

    def test_key_logical_operators(self):
        """Test _Key logical operators."""
        key1 = _Key("flag1")
        key2 = _Key("flag2")
        
        # And
        op_and = key1 & key2
        assert isinstance(op_and, Operator)
        
        # Or
        op_or = key1 | key2
        assert isinstance(op_or, Operator)


class TestOperator:
    """Tests for Operator class."""

    def test_operator_initialization_binary(self):
        """Test Operator initialization with binary operation."""
        key = _Key("value")
        op = Operator(operator.eq, key, 5)
        assert op._op == operator.eq
        assert op._a == key
        assert op._b == 5

    def test_operator_initialization_unary(self):
        """Test Operator initialization with unary operation."""
        key = _Key("value")
        op = Operator(operator.not_, key)
        assert op._op == operator.not_
        assert op._a == key
        assert op._b is None

    def test_operator_evaluate_equality(self):
        """Test Operator evaluation with equality."""
        key = _Key("age")
        op = key == 30
        
        # Test with matching value
        item = {"age": 30}
        assert op.evaluate(item) is True
        
        # Test with non-matching value
        item = {"age": 25}
        assert op.evaluate(item) is False

    def test_operator_evaluate_comparison(self):
        """Test Operator evaluation with comparison operators."""
        key = _Key("score")
        
        # Greater than
        op_gt = key > 80
        assert op_gt.evaluate({"score": 90}) is True
        assert op_gt.evaluate({"score": 70}) is False
        
        # Less than
        op_lt = key < 50
        assert op_lt.evaluate({"score": 40}) is True
        assert op_lt.evaluate({"score": 60}) is False

    def test_operator_evaluate_nested(self):
        """Test Operator evaluation with nested operators."""
        key = _Key("value")
        # (value + 5) > 10
        op = (key + 5) > 10
        
        assert op.evaluate({"value": 6}) is True  # 6 + 5 = 11 > 10
        assert op.evaluate({"value": 5}) is False  # 5 + 5 = 10 not > 10
        assert op.evaluate({"value": 4}) is False  # 4 + 5 = 9 < 10

    def test_operator_repr(self):
        """Test Operator string representation."""
        key = _Key("age")
        op = key == 30
        repr_str = repr(op)
        assert "Operation" in repr_str
        assert "eq" in repr_str

    def test_operator_chaining(self):
        """Test chaining multiple operators."""
        key = _Key("x")
        # ((x + 5) * 2) > 20
        op = ((key + 5) * 2) > 20
        
        assert op.evaluate({"x": 6}) is True  # (6+5)*2 = 22 > 20
        assert op.evaluate({"x": 5}) is False  # (5+5)*2 = 20 not > 20


class TestFilter:
    """Tests for Filter class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40],
            "score": [85, 90, 75, 95],
            "tags": [["new"], ["old", "special"], ["new"], ["old"]]
        })

    def test_filter_initialization_empty(self):
        """Test Filter initialization with no operators."""
        f = Filter()
        assert len(f._ops) == 0
        assert len(f._tags) == 0

    def test_filter_initialization_with_operators(self):
        """Test Filter initialization with operators."""
        key = _Key("age")
        op = key > 30
        f = Filter(op)
        assert len(f._ops) == 1
        assert f._ops[0] == op

    def test_filter_initialization_with_tags(self):
        """Test Filter initialization with tags."""
        f = Filter(tags=["tag1", "tag2"])
        assert f._tags == {"tag1", "tag2"}

    def test_filter_apply_single_operator(self, sample_df):
        """Test Filter.apply with a single operator."""
        key = _Key("age")
        f = Filter(key > 30)
        result = f.apply(sample_df)
        
        # Should return rows where age > 30
        assert len(result) == 2
        assert list(result["name"]) == ["Charlie", "David"]

    def test_filter_apply_multiple_operators(self, sample_df):
        """Test Filter.apply with multiple operators (AND logic)."""
        age_key = _Key("age")
        score_key = _Key("score")
        f = Filter(age_key > 25, score_key >= 90)
        result = f.apply(sample_df)
        
        # Should return rows where age > 25 AND score >= 90
        assert len(result) == 2
        assert list(result["name"]) == ["Bob", "David"]

    def test_filter_apply_with_tags(self, sample_df):
        """Test Filter.apply with tag filtering."""
        f = Filter(tags=["old"])
        result = f.apply(sample_df)
        
        # Should return rows with "old" tag
        assert len(result) == 2
        assert set(result["name"]) == {"Bob", "David"}

    def test_filter_apply_operators_and_tags(self, sample_df):
        """Test Filter.apply with both operators and tags."""
        age_key = _Key("age")
        f = Filter(age_key > 25, tags=["old"])
        result = f.apply(sample_df)
        
        # Should return rows where age > 25 AND has "old" tag
        assert len(result) == 2
        assert set(result["name"]) == {"Bob", "David"}

    def test_filter_and_operator(self):
        """Test Filter.__and__ method."""
        key = _Key("age")
        f1 = Filter(key > 20)
        f2 = Filter(key < 40)
        combined = f1 & f2
        
        assert len(combined._ops) == 2
        assert isinstance(combined, Filter)

    def test_filter_and_with_none(self):
        """Test Filter.__and__ with None."""
        key = _Key("age")
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
        key = _Key("age")
        f = Filter(key > 20, key < 40)
        repr_str = repr(f)
        assert "Filter" in repr_str

    def test_filter_empty_dataframe(self):
        """Test Filter on empty DataFrame."""
        empty_df = pd.DataFrame(columns=["age", "name"])
        key = _Key("age")
        f = Filter(key > 20)
        result = f.apply(empty_df)
        
        assert len(result) == 0
        assert list(result.columns) == ["age", "name"]


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
        return pd.DataFrame({
            "name": ["Charlie", "Alice", "David", "Bob"],
            "age": [35, 25, 40, 30],
            "score": [75, 85, 95, 90]
        })

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

    def test_sorter_apply_single_key_ascending(self, sample_df):
        """Test Sorter.apply with single key, ascending."""
        instr = SortInstruction("age", ascending=True)
        s = Sorter(instr)
        result = s.apply(sample_df)
        
        # Should be sorted by age ascending
        assert list(result["name"]) == ["Alice", "Bob", "Charlie", "David"]
        assert list(result["age"]) == [25, 30, 35, 40]

    def test_sorter_apply_single_key_descending(self, sample_df):
        """Test Sorter.apply with single key, descending."""
        instr = SortInstruction("score", ascending=False)
        s = Sorter(instr)
        result = s.apply(sample_df)
        
        # Should be sorted by score descending
        assert list(result["name"]) == ["David", "Bob", "Alice", "Charlie"]
        assert list(result["score"]) == [95, 90, 85, 75]

    def test_sorter_apply_multiple_keys(self, sample_df):
        """Test Sorter.apply with multiple sort keys."""
        # Add duplicate ages for testing
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 25, 30, 30],
            "score": [85, 90, 75, 95]
        })
        
        instr1 = SortInstruction("age", ascending=True)
        instr2 = SortInstruction("score", ascending=False)
        s = Sorter(instr1, instr2)
        result = s.apply(df)
        
        # Should sort by age first, then by score descending
        assert list(result["name"]) == ["Bob", "Alice", "David", "Charlie"]

    def test_sorter_apply_empty_sorter(self, sample_df):
        """Test Sorter.apply with no instructions returns unchanged DataFrame."""
        s = Sorter()
        result = s.apply(sample_df)
        
        # Should return DataFrame unchanged
        pd.testing.assert_frame_equal(result, sample_df)

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

    def test_sorter_empty_dataframe(self):
        """Test Sorter on empty DataFrame."""
        empty_df = pd.DataFrame(columns=["age", "name", "score"])
        instr = SortInstruction("age")
        s = Sorter(instr)
        result = s.apply(empty_df)
        
        assert len(result) == 0
        assert list(result.columns) == ["age", "name", "score"]


class TestIntegration:
    """Integration tests for Filter and Sorter together."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 28],
            "score": [85, 90, 75, 95, 88],
            "tags": [["new"], ["old"], ["new", "special"], ["old"], ["new"]]
        })

    def test_filter_then_sort(self, sample_df):
        """Test applying filter then sort."""
        # Filter: age > 27
        age_key = _Key("age")
        f = Filter(age_key > 27)
        filtered_df = f.apply(sample_df)
        
        # Sort by score descending
        instr = SortInstruction("score", ascending=False)
        s = Sorter(instr)
        result = s.apply(filtered_df)
        
        # Should have 4 people (age > 27), sorted by score descending
        assert len(result) == 4
        assert list(result["name"]) == ["David", "Bob", "Eve", "Charlie"]

    def test_complex_filter_with_sort(self, sample_df):
        """Test complex filter with multiple conditions and sorting."""
        age_key = _Key("age")
        score_key = _Key("score")
        
        # Filter: age >= 28 AND score > 80
        f = Filter(age_key >= 28, score_key > 80)
        filtered_df = f.apply(sample_df)
        
        # Sort by age ascending
        s = Sorter(SortInstruction("age"))
        result = s.apply(filtered_df)
        
        # Should have Eve, Bob, David sorted by age
        assert len(result) == 3
        assert list(result["name"]) == ["Eve", "Bob", "David"]
        assert list(result["age"]) == [28, 30, 40]
