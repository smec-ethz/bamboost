"""Tests for bamboost._config module."""

from bamboost import _config


class TestConfigClass:
    """Tests for _Config main configuration class."""

    def test_config_initialization(self):
        """Test _Config initialization with defaults."""
        cfg = _config._Config()

        assert cfg.mpi is False
        assert cfg.log_file_lock_severity == "WARNING"
        assert cfg.sort_table_key == "created_at"

    def test_config_flat_update(self):
        """Test that dictionary updates work correctly by flattening 'options'."""
        cfg = _config._Config()
        cfg.mpi = True
        assert cfg.mpi is True


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


class TestGlobalConfig:
    """Tests for global config instance."""

    def test_global_config_exists(self):
        """Test that global config instance exists."""
        assert _config.config is not None
        assert isinstance(_config.config, _config._Config)
