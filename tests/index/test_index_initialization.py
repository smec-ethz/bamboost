import sqlite3

from sqlalchemy import Column, String, create_engine

from bamboost.index.base import Index
from bamboost.index.schema import (
    _column_sql_for_alter,
    collections_table,
    parameters_table,
    simulations_table,
)


def test_index_initialization_in_memory():
    index = Index(sql_file=":memory:")
    assert isinstance(index, Index)


def test_lazy_default_index_exists():
    assert isinstance(Index.default, Index)
    assert Index.default is Index.default  # Same instance (singleton)


def test_index_initialization_adds_all_missing_columns(tmp_path):
    db_file = tmp_path / "legacy.sqlite"

    with sqlite3.connect(db_file) as connection:
        # Intentionally create legacy/partial schemas with missing columns.
        connection.execute(
            """
            CREATE TABLE collections (
                uid TEXT PRIMARY KEY,
                path TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_uid TEXT,
                name TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                modified_at DATETIME NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER NOT NULL
            )
            """
        )
        connection.commit()

    _ = Index(sql_file=db_file)

    expected_by_table = {
        collections_table.name: set(collections_table.c.keys()),
        simulations_table.name: set(simulations_table.c.keys()),
        parameters_table.name: set(parameters_table.c.keys()),
    }

    with sqlite3.connect(db_file) as connection:
        for table_name, expected_columns in expected_by_table.items():
            rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
            actual_columns = {row[1] for row in rows}
            assert expected_columns.issubset(actual_columns)


def test_column_sql_for_alter_non_null_without_server_default_is_permissive():
    engine = create_engine("sqlite:///:memory:")
    column = Column("required_name", String, nullable=False)

    sql = _column_sql_for_alter(engine, column)

    assert sql.startswith("required_name")
    assert "NOT NULL" not in sql
    assert "DEFAULT" not in sql
