"""Tests for bamboost.core.utilities module."""

from pathlib import Path

import pytest

from bamboost.core.utilities import (
    FilePicker,
    dedupe_str_iter,
    flatten_dict,
    to_camel_case,
    tree,
    unflatten_dict,
)


class TestFlattenDict:
    """Tests for flatten_dict function."""

    def test_flatten_dict_simple(self):
        """Test flattening a simple nested dictionary."""
        nested = {"a": {"b": 1, "c": 2}}
        result = flatten_dict(nested)
        assert result == {"a.b": 1, "a.c": 2}

    def test_flatten_dict_deeply_nested(self):
        """Test flattening a deeply nested dictionary."""
        nested = {"a": {"b": {"c": {"d": 1}}}}
        result = flatten_dict(nested)
        assert result == {"a.b.c.d": 1}

    def test_flatten_dict_mixed_depth(self):
        """Test flattening a dictionary with mixed nesting depths."""
        nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        result = flatten_dict(nested)
        assert result == {"a": 1, "b.c": 2, "b.d.e": 3}

    def test_flatten_dict_empty(self):
        """Test flattening an empty dictionary."""
        result = flatten_dict({})
        assert result == {}

    def test_flatten_dict_no_nesting(self):
        """Test flattening a dictionary with no nesting."""
        flat = {"a": 1, "b": 2, "c": 3}
        result = flatten_dict(flat)
        assert result == flat

    def test_flatten_dict_custom_separator(self):
        """Test flattening with a custom separator."""
        nested = {"a": {"b": 1}}
        result = flatten_dict(nested, seperator="/")
        assert result == {"a/b": 1}

    def test_flatten_dict_with_lists(self):
        """Test that lists are preserved (not flattened)."""
        nested = {"a": {"b": [1, 2, 3]}}
        result = flatten_dict(nested)
        assert result == {"a.b": [1, 2, 3]}


class TestUnflattenDict:
    """Tests for unflatten_dict function."""

    def test_unflatten_dict_simple(self):
        """Test unflattening a simple flat dictionary."""
        flat = {"a.b": 1, "a.c": 2}
        result = unflatten_dict(flat)
        assert result == {"a": {"b": 1, "c": 2}}

    def test_unflatten_dict_deeply_nested(self):
        """Test unflattening a deeply nested flat dictionary."""
        flat = {"a.b.c.d": 1}
        result = unflatten_dict(flat)
        assert result == {"a": {"b": {"c": {"d": 1}}}}

    def test_unflatten_dict_mixed_depth(self):
        """Test unflattening a dictionary with mixed depths."""
        flat = {"a": 1, "b.c": 2, "b.d.e": 3}
        result = unflatten_dict(flat)
        assert result == {"a": 1, "b": {"c": 2, "d": {"e": 3}}}

    def test_unflatten_dict_empty(self):
        """Test unflattening an empty dictionary."""
        result = unflatten_dict({})
        assert result == {}

    def test_unflatten_dict_no_dots(self):
        """Test unflattening a dictionary with no dots."""
        flat = {"a": 1, "b": 2}
        result = unflatten_dict(flat)
        assert result == flat

    def test_unflatten_dict_custom_separator(self):
        """Test unflattening with a custom separator."""
        flat = {"a/b": 1}
        result = unflatten_dict(flat, seperator="/")
        assert result == {"a": {"b": 1}}

    def test_flatten_unflatten_roundtrip(self):
        """Test that flatten and unflatten are inverse operations."""
        original = {"a": {"b": {"c": 1}}, "d": {"e": 2, "f": 3}}
        flattened = flatten_dict(original)
        unflattened = unflatten_dict(flattened)
        assert unflattened == original


class TestTree:
    """Tests for tree function."""

    def test_tree_empty_directory(self, tmp_path):
        """Test tree with an empty directory."""
        result = tree(tmp_path)
        assert tmp_path.name in result
        assert "0 directories" in result

    def test_tree_with_files(self, tmp_path):
        """Test tree with files."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        result = tree(tmp_path)
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "2 files" in result

    def test_tree_with_subdirectories(self, tmp_path):
        """Test tree with subdirectories."""
        (tmp_path / "subdir1").mkdir()
        (tmp_path / "subdir2").mkdir()
        result = tree(tmp_path)
        assert "subdir1" in result
        assert "subdir2" in result
        assert "2 directories" in result

    def test_tree_nested_structure(self, tmp_path):
        """Test tree with nested structure."""
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "file1.txt").touch()
        (tmp_path / "dir1" / "subdir").mkdir()
        result = tree(tmp_path)
        assert "dir1" in result
        assert "file1.txt" in result
        assert "subdir" in result

    def test_tree_limit_to_directories(self, tmp_path):
        """Test tree with limit_to_directories flag."""
        (tmp_path / "file.txt").touch()
        (tmp_path / "subdir").mkdir()
        result = tree(tmp_path, limit_to_directories=True)
        assert "subdir" in result
        assert "file.txt" not in result

    def test_tree_level_limit(self, tmp_path):
        """Test tree with level limit."""
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "dir2").mkdir()
        (tmp_path / "dir1" / "dir2" / "dir3").mkdir()
        
        result = tree(tmp_path, level=1)
        assert "dir1" in result
        # dir2 should not appear as we limited to level 1
        assert "dir2" not in result

    def test_tree_string_path(self, tmp_path):
        """Test tree accepts string path."""
        result = tree(str(tmp_path))
        assert tmp_path.name in result


class TestFilePicker:
    """Tests for FilePicker class."""

    def test_filepicker_initialization(self, tmp_path):
        """Test FilePicker initialization."""
        picker = FilePicker(tmp_path)
        assert picker.path == tmp_path

    def test_filepicker_empty_directory(self, tmp_path):
        """Test FilePicker with empty directory."""
        picker = FilePicker(tmp_path)
        assert len(picker._dict) == 0

    def test_filepicker_with_files(self, tmp_path):
        """Test FilePicker with files."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        picker = FilePicker(tmp_path)
        
        assert "file1.txt" in picker._dict
        assert "file2.txt" in picker._dict
        assert picker["file1.txt"] == (tmp_path / "file1.txt").absolute()

    def test_filepicker_with_subdirectories(self, tmp_path):
        """Test FilePicker with subdirectories."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file.txt").touch()
        picker = FilePicker(tmp_path)
        
        assert "subdir/file.txt" in picker._dict
        assert picker["subdir/file.txt"] == (tmp_path / "subdir" / "file.txt").absolute()

    def test_filepicker_nested_structure(self, tmp_path):
        """Test FilePicker with nested directory structure."""
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "dir2").mkdir()
        (tmp_path / "dir1" / "dir2" / "file.txt").touch()
        picker = FilePicker(tmp_path)
        
        assert "dir1/dir2/file.txt" in picker._dict

    def test_filepicker_getitem(self, tmp_path):
        """Test FilePicker __getitem__ method."""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        picker = FilePicker(tmp_path)
        
        result = picker["test.txt"]
        assert result == file_path.absolute()

    def test_filepicker_str(self, tmp_path):
        """Test FilePicker __str__ method returns tree."""
        (tmp_path / "file.txt").touch()
        picker = FilePicker(tmp_path)
        result = str(picker)
        assert "file.txt" in result

    def test_filepicker_with_file_path(self, tmp_path):
        """Test FilePicker with a file path (not directory)."""
        file_path = tmp_path / "file.txt"
        file_path.touch()
        picker = FilePicker(file_path)
        # Should have empty dict since it's not a directory
        assert len(picker._dict) == 0


class TestToCamelCase:
    """Tests for to_camel_case function."""

    def test_to_camel_case_simple(self):
        """Test converting simple string to camel case."""
        result = to_camel_case("hello world")
        assert result == "helloWorld"

    def test_to_camel_case_single_word(self):
        """Test converting single word."""
        result = to_camel_case("hello")
        assert result == "hello"

    def test_to_camel_case_multiple_words(self):
        """Test converting multiple words."""
        result = to_camel_case("this is a test")
        assert result == "thisIsATest"

    def test_to_camel_case_already_camel(self):
        """Test with already camel-cased string."""
        result = to_camel_case("camelCase")
        assert result == "camelcase"  # First word lowercased

    def test_to_camel_case_empty_string(self):
        """Test with empty string."""
        result = to_camel_case("")
        assert result == ""


class TestDedupeStrIter:
    """Tests for dedupe_str_iter function."""

    def test_dedupe_str_iter_list(self):
        """Test deduplication of a list of strings."""
        result = dedupe_str_iter(["a", "b", "a", "c"])
        assert result == {"a", "b", "c"}

    def test_dedupe_str_iter_removes_empty(self):
        """Test that empty strings are removed."""
        result = dedupe_str_iter(["a", "", "b", None, "c"])
        assert result == {"a", "b", "c"}

    def test_dedupe_str_iter_single_string(self):
        """Test with a single string (not iterable)."""
        result = dedupe_str_iter("test")
        assert result == {"test"}

    def test_dedupe_str_iter_none(self):
        """Test with None input."""
        result = dedupe_str_iter(None)
        assert result == set()

    def test_dedupe_str_iter_empty_list(self):
        """Test with empty list."""
        result = dedupe_str_iter([])
        assert result == set()

    def test_dedupe_str_iter_all_empty(self):
        """Test with all empty strings."""
        result = dedupe_str_iter(["", "", ""])
        assert result == set()

    def test_dedupe_str_iter_tuple(self):
        """Test with tuple input."""
        result = dedupe_str_iter(("a", "b", "a"))
        assert result == {"a", "b"}

    def test_dedupe_str_iter_set(self):
        """Test with set input (already deduplicated)."""
        result = dedupe_str_iter({"a", "b", "c"})
        assert result == {"a", "b", "c"}
