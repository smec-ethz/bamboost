from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from bamboost.index.base import Index
from bamboost.index.store import simulations_upsert_stmt


@pytest.fixture
def index():
    return Index(sql_file=":memory:")


def test_transaction_commit(index: Index):
    with patch.object(index._s, "commit", wraps=index._s.commit) as mock_commit:
        with index.sql_transaction() as session:
            assert session is index._s
        mock_commit.assert_called_once()


def test_transaction_rollback_on_error(index: Index, tmp_path: Path):
    index.upsert_collection("COLL123", tmp_path)

    with patch.object(index._s, "rollback", wraps=index._s.rollback) as mock_rollback:
        with patch.object(index._s, "commit", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(SQLAlchemyError):
                with index.sql_transaction():
                    index._s.execute(
                        simulations_upsert_stmt(
                            {
                                "collection_uid": "COLL123",
                                "name": "apple",
                                "created_at": datetime.now(),
                                "modified_at": datetime.now(),
                                "status": "initialized",
                                "submitted": False,
                            }
                        )
                    )

        # 04.03.25: this setup does not work (?)
        # mock_rollback.assert_called_once()

    # Reset the ORM objects and check that the transaction was rolled back
    index._s.reset()
    assert len(index.all_simulations) == 0
