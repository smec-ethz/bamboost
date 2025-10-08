from bamboost import _config


def test_local_dir_created_for_database(tmp_path, monkeypatch):
    local_dir = tmp_path / "local-dir"
    monkeypatch.setattr(_config, "LOCAL_DIR", local_dir)

    assert not local_dir.exists()

    index_config = _config._IndexOptions.from_dict({})

    assert local_dir.exists()
    assert index_config.databaseFile == local_dir / _config.DATABASE_FILE_NAME
