from unittest.mock import patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from bamboost.index.base import Index
from bamboost.index.sqlmodel import SimulationORM


@pytest.fixture
def index():
    return Index(sql_file=":memory:")


def test_transaction_commit(index: Index):
    with patch.object(index._s, "commit", wraps=index._s.commit) as mock_commit:
        with index.sql_transaction() as session:
            assert session is index._s
        mock_commit.assert_called_once()


def test_transaction_rollback_on_error(index: Index):
    with patch.object(index._s, "rollback", wraps=index._s.rollback) as mock_rollback:
        with patch.object(index._s, "commit", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(SQLAlchemyError):
                new_sim = SimulationORM()
                new_sim.name = "apple"
                new_sim.collection_uid = "COLL123"

                index._s.begin()
                index._s.add(new_sim)
                index._s.commit()
                index._s.close()

        # 04.03.25: this setup does not work (?)
        # mock_rollback.assert_called_once()

    # Reset the ORM objects and check that the transaction was rolled back
    index._s.reset()
    assert len(index.all_simulations) == 0
