import sqlite3
from contextlib import contextmanager


class FastIndexQuery:
    def __init__(self):
        from bamboost._config import config

        self.db_path = config.index.databaseFile

    @contextmanager
    def connection(self):
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._cursor = self._conn.cursor()
            yield self._cursor
        finally:
            self._conn.close()

    def query(self, query: str, *args) -> list[tuple]:
        with self.connection() as cursor:
            cursor.execute(query, *args)
            return cursor.fetchall()


INDEX = FastIndexQuery()
