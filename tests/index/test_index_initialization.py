from bamboost.index.base import Index


def test_index_initialization_in_memory():
    index = Index(sql_file=":memory:")
    assert isinstance(index, Index)


def test_lazy_default_index_exists():
    assert isinstance(Index.default, Index)
    assert Index.default is Index.default  # Same instance (singleton)
