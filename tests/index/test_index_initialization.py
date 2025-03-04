from bamboost.index.base import Index


def test_index_initialization_in_memory():
    index = Index(sql_file=":memory:")
    assert isinstance(index, Index)


def test_lazy_default_index_exists():
    assert isinstance(Index.default, Index)
    assert Index.default is Index.default  # Same instance (singleton)


def test_index_comm_self(tmp_path):
    index = Index(sql_file=tmp_path / "test.db")

    with index.comm_self():
        assert index._comm.Get_size() == 1  # COMM_SELF should only have 1 rank
