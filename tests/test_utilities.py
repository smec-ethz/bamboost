"""Tests for bamboost.utilities module."""

from pathlib import Path

import pytest

from bamboost.utilities import PathSet, full_class_name


class TestPathSet:
    """Tests for PathSet class."""

    def test_pathset_initialization_empty(self):
        """Test PathSet initialization with no arguments."""
        ps = PathSet()
        assert len(ps) == 0
        assert isinstance(ps, set)

    def test_pathset_initialization_with_strings(self, tmp_path):
        """Test PathSet initialization with string paths."""
        paths = [str(tmp_path / "file1.txt"), str(tmp_path / "file2.txt")]
        ps = PathSet(paths)
        assert len(ps) == 2
        # All items should be resolved Path objects
        for item in ps:
            assert isinstance(item, Path)
            assert item.is_absolute()

    def test_pathset_initialization_with_paths(self, tmp_path):
        """Test PathSet initialization with Path objects."""
        paths = [tmp_path / "file1.txt", tmp_path / "file2.txt"]
        ps = PathSet(paths)
        assert len(ps) == 2
        for item in ps:
            assert isinstance(item, Path)
            assert item.is_absolute()

    def test_pathset_resolves_paths(self, tmp_path):
        """Test that PathSet resolves relative paths."""
        # Use relative path notation
        relative_path = "."
        ps = PathSet([relative_path])
        assert len(ps) == 1
        item = next(iter(ps))
        assert item.is_absolute()

    def test_pathset_expands_user(self, tmp_path):
        """Test that PathSet expands user home directory."""
        # Note: This test assumes ~ expansion works
        path_with_tilde = "~/test_file.txt"
        ps = PathSet([path_with_tilde])
        assert len(ps) == 1
        item = next(iter(ps))
        assert "~" not in str(item)
        assert item.is_absolute()

    def test_pathset_add_string(self, tmp_path):
        """Test adding a string path to PathSet."""
        ps = PathSet()
        ps.add(str(tmp_path / "file.txt"))
        assert len(ps) == 1
        item = next(iter(ps))
        assert isinstance(item, Path)
        assert item.is_absolute()

    def test_pathset_add_path(self, tmp_path):
        """Test adding a Path object to PathSet."""
        ps = PathSet()
        ps.add(tmp_path / "file.txt")
        assert len(ps) == 1
        item = next(iter(ps))
        assert isinstance(item, Path)
        assert item.is_absolute()

    def test_pathset_duplicates_removed(self, tmp_path):
        """Test that PathSet removes duplicate paths."""
        path = tmp_path / "file.txt"
        ps = PathSet([path, path, str(path)])
        assert len(ps) == 1

    def test_pathset_none_iterable(self):
        """Test PathSet with None iterable."""
        ps = PathSet(None)
        assert len(ps) == 0


class TestFullClassName:
    """Tests for full_class_name function."""

    def test_full_class_name_regular_class(self):
        """Test full_class_name with a regular class."""
        
        class TestClass:
            pass
        
        result = full_class_name(TestClass)
        assert "TestClass" in result
        assert "test_utilities" in result

    def test_full_class_name_builtin_class(self):
        """Test full_class_name with a builtin class."""
        result = full_class_name(int)
        # Builtin classes should return just their qualname
        assert result == "int"

    def test_full_class_name_with_module(self):
        """Test full_class_name includes module name."""
        from bamboost.utilities import PathSet
        
        result = full_class_name(PathSet)
        assert result == "bamboost.utilities.PathSet"

    def test_full_class_name_nested_class(self):
        """Test full_class_name with a nested class."""
        
        class OuterClass:
            class InnerClass:
                pass
        
        result = full_class_name(OuterClass.InnerClass)
        assert "OuterClass.InnerClass" in result

    def test_full_class_name_none_module(self):
        """Test full_class_name when module is None."""
        # Create a class with no module (edge case)
        class NoModuleClass:
            pass
        
        # Manually set module to None to test edge case
        NoModuleClass.__module__ = None
        result = full_class_name(NoModuleClass)
        assert result == "NoModuleClass"
