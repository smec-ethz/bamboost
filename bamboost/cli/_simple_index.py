import sqlite3
from contextlib import contextmanager

from bamboost._typing import StrPath


class FastIndexQuery:
    def __init__(self, db_path: StrPath):
        self.db_path = db_path

    @contextmanager
    def connection(self):
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._cursor = self._conn.cursor()
            yield self._cursor
        finally:
            self._conn.close()

    def query(self, query: str) -> list[tuple]:
        with self.connection() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
