from bamboost.tui.common import Caller
from bamboost.tui.pages.db import Database as _Database


def _get_database_from_caller() -> _Database:
    widget_stack = Caller.widget_stack
    for widget in reversed(widget_stack):
        if isinstance(widget, _Database):
            return widget
    raise ValueError("No Database instance found in the widget stack")


def get_entry_in_focus():
    """Returns the bamboost Simulation object in focus."""
    cli_db: _Database = _get_database_from_caller()
    uid = cli_db.table._get_entry_in_focus()["id"]
    return cli_db.db[uid]


class Database:
    def __init__(self):
        # self._self: _Database = _get_instance_by_class(_Database)[0]
        self._self: _Database = _get_database_from_caller()

    def __getattr__(self, name):
        return getattr(self._self, name)

    def uid_in_focus(self) -> str:
        uid = self._self.table._get_entry_in_focus()["id"]
        return uid

    def sim_in_focus(self):
        return self._self.db[self.uid_in_focus()]

    def print(self):
        print(self._self)
