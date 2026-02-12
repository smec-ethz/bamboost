"""Tests for bamboost._config module."""

from pathlib import Path

import pytest

from bamboost import _config
from bamboost.constants import DEFAULT_DATABASE_FILE_NAME
from bamboost.utilities import PathSet


class TestIndexOptions:
    """Tests for _IndexOptions configuration."""

    def test_local_dir_created_for_database(self, tmp_path, monkeypatch):
        """Test that local directory is created when loading index config."""
        local_dir = tmp_path / "local-dir"
        monkeypatch.setattr(_config, "LOCAL_DIR", local_dir)

        assert not local_dir.exists()

        index_config = _config._IndexOptions.from_dict({})

        assert local_dir.exists()
        assert index_config.databaseFile == local_dir / DEFAULT_DATABASE_FILE_NAME

    def test_index_options_default_values(self, tmp_path, monkeypatch):
        """Test _IndexOptions default values."""
        local_dir = tmp_path / "local-dir"
        monkeypatch.setattr(_config, "LOCAL_DIR", local_dir)
        
        index_config = _config._IndexOptions.from_dict({})
        
        assert isinstance(index_config.databaseFile, Path)
        assert index_config.databaseFile.name == DEFAULT_DATABASE_FILE_NAME

    def test_index_options_custom_database_file(self, tmp_path, monkeypatch):
        """Test _IndexOptions with custom database file."""
        local_dir = tmp_path / "local-dir"
        monkeypatch.setattr(_config, "LOCAL_DIR", local_dir)
        
        custom_db = tmp_path / "custom.db"
        index_config = _config._IndexOptions.from_dict({"databaseFile": str(custom_db)})
        
        assert index_config.databaseFile == custom_db


class TestPathsOptions:
    """Tests for _PathsOptions configuration."""

    def test_paths_options_default(self):
        """Test _PathsOptions default initialization."""
        paths = _config._PathsOptions()
        
        assert isinstance(paths.localDir, Path)
        assert isinstance(paths.cacheDir, Path)
        assert isinstance(paths.excluded, PathSet)

    def test_paths_options_from_dict(self, tmp_path):
        """Test _PathsOptions.from_dict with custom values."""
        custom_local = tmp_path / "custom_local"
        custom_cache = tmp_path / "custom_cache"
        
        config_dict = {
            "localDir": str(custom_local),
            "cacheDir": str(custom_cache),
            "excluded": [str(tmp_path / "exclude1"), str(tmp_path / "exclude2")]
        }
        
        paths = _config._PathsOptions.from_dict(config_dict)
        
        assert paths.localDir == custom_local
        assert paths.cacheDir == custom_cache
        assert len(paths.excluded) == 2

    def test_paths_options_excluded_as_pathset(self, tmp_path):
        """Test that excluded paths are properly converted to PathSet."""
        paths = _config._PathsOptions.from_dict({
            "excluded": [str(tmp_path / "dir1"), str(tmp_path / "dir2")]
        })
        
        assert isinstance(paths.excluded, _config.PathSet)
        assert len(paths.excluded) == 2


class TestGeneralOptions:
    """Tests for _GeneralOptions configuration."""

    def test_general_options_default(self):
        """Test _GeneralOptions default values."""
        options = _config._GeneralOptions()
        
        assert options.mpi is False

    def test_general_options_from_dict(self):
        """Test _GeneralOptions.from_dict."""
        options = _config._GeneralOptions.from_dict({"mpi": True})
        
        assert options.mpi is True

    def test_general_options_from_dict_empty(self):
        """Test _GeneralOptions.from_dict with empty dict."""
        options = _config._GeneralOptions.from_dict({})
        
        assert options.mpi is False


class TestConfigClass:
    """Tests for _Config main configuration class."""

    def test_config_initialization(self):
        """Test _Config initialization with defaults."""
        cfg = _config._Config()
        
        assert isinstance(cfg.paths, _config._PathsOptions)
        assert isinstance(cfg.options, _config._GeneralOptions)
        assert isinstance(cfg.index, _config._IndexOptions)

    def test_config_from_dict(self, tmp_path, monkeypatch):
        """Test _Config.from_dict with nested configuration."""
        local_dir = tmp_path / "local"
        monkeypatch.setattr(_config, "LOCAL_DIR", local_dir)
        
        config_dict = {
            "paths": {
                "localDir": str(tmp_path / "custom_local")
            },
            "options": {
                "mpi": True
            }
        }
        
        cfg = _config._Config.from_dict(config_dict)
        
        assert cfg.paths.localDir == tmp_path / "custom_local"
        assert cfg.options.mpi is True

    def test_config_nested_update(self):
        """Test that nested dictionary updates work correctly."""
        dict1 = {"a": {"b": 1, "c": 2}}
        dict2 = {"a": {"c": 3, "d": 4}}
        
        _config._nested_dict_update(dict1, dict2)
        
        assert dict1 == {"a": {"b": 1, "c": 3, "d": 4}}


class TestConfigUtilities:
    """Tests for configuration utility functions."""

    def test_find_root_dir_not_found(self, tmp_path, monkeypatch):
        """Test _find_root_dir when no anchor files exist."""
        # Change to a temp directory with no anchor files
        monkeypatch.chdir(tmp_path)
        
        result = _config._find_root_dir()
        
        # Should return None when no anchor found
        assert result is None

    def test_find_root_dir_with_git(self, tmp_path, monkeypatch):
        """Test _find_root_dir when .git directory exists."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        
        subdir = tmp_path / "subdir" / "deep"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        
        result = _config._find_root_dir()
        
        # Should find the directory containing .git
        assert result == tmp_path

    def test_find_root_dir_with_pyproject(self, tmp_path, monkeypatch):
        """Test _find_root_dir when pyproject.toml exists."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.touch()
        
        subdir = tmp_path / "src"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        
        result = _config._find_root_dir()
        
        assert result == tmp_path

    def test_nested_dict_update_simple(self):
        """Test _nested_dict_update with simple dictionaries."""
        target = {"a": 1, "b": 2}
        source = {"b": 3, "c": 4}
        
        _config._nested_dict_update(target, source)
        
        assert target == {"a": 1, "b": 3, "c": 4}

    def test_nested_dict_update_deep(self):
        """Test _nested_dict_update with deeply nested dictionaries."""
        target = {"a": {"b": {"c": 1}}}
        source = {"a": {"b": {"d": 2}, "e": 3}}
        
        _config._nested_dict_update(target, source)
        
        assert target == {"a": {"b": {"c": 1, "d": 2}, "e": 3}}

    def test_nested_dict_update_mixed_types(self):
        """Test _nested_dict_update with mixed value types."""
        target = {"a": {"b": 1}, "c": [1, 2]}
        source = {"a": {"d": 2}, "c": [3, 4]}
        
        _config._nested_dict_update(target, source)
        
        # Non-dict values should be replaced
        assert target == {"a": {"b": 1, "d": 2}, "c": [3, 4]}


class TestGlobalConfig:
    """Tests for global config instance."""

    def test_global_config_exists(self):
        """Test that global config instance exists."""
        assert _config.config is not None
        assert isinstance(_config.config, _config._Config)

    def test_global_config_has_paths(self):
        """Test that global config has paths configuration."""
        assert hasattr(_config.config, "paths")
        assert isinstance(_config.config.paths, _config._PathsOptions)

    def test_global_config_has_options(self):
        """Test that global config has options configuration."""
        assert hasattr(_config.config, "options")
        assert isinstance(_config.config.options, _config._GeneralOptions)

    def test_global_config_has_index(self):
        """Test that global config has index configuration."""
        assert hasattr(_config.config, "index")
        assert isinstance(_config.config.index, _config._IndexOptions)
