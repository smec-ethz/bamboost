from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime
from typing import Union

import urwid
from bamboost.tui.common import Caller, Spinner
from bamboost.tui.pages.db import Database
from bamboost.tui.widgets.custom_widgets import (
    cActionItem,
    cEdit,
    cListBoxSelectionCharacter,
    cPopup,
)
from bamboost.tui.widgets.rounded_line_box import cRoundedLineBox
from fuzzywuzzy import fuzz

import bamboost.core as bb
from bamboost.core.extensions.remote_manager import Remote


class Index(urwid.Frame):
    """Index widget for displaying a list of known databases.

    Arguments:
        - data (dict): A dictionary containing keys as unique identifiers and
          values as corresponding paths.

    Attributes:
        - data (dict): A dictionary containing keys as unique identifiers and
          values as corresponding paths.
        - headers (urwid.Columns): A widget containing the headers for the
          table.
        - table (CustomListBox): A list box widget containing the data.
        - header (urwid.Pile): A pile widget for the headers with spacing.
        - footer (urwid.Filler): A filler widget containing the search bar.
    """

    signals = ["set_footer"]

    def __init__(self, data: dict) -> None:
        self._initialize(data)
        super().__init__(
            urwid.Padding(self.table, left=2, right=2),
            urwid.Padding(self.header, left=2, right=2),
            self.footer,
            focus_part="footer",
        )

    def _initialize(self, data: dict) -> None:
        # Get the last modified time of each databasse
        mtime = {key: os.path.getmtime(path) for key, path in data.items()}

        self.data = {}
        self.data = self._process_data(data, mtime)

        def bold_text(text, *args, **kwargs):
            return urwid.AttrWrap(urwid.Text(text, *args, **kwargs), "bold")

        self.table = cListBoxSelectionCharacter(
            [self._create_item(key) for key in self.data],
            {0: "5", 1: "default", 2: "6", 3: "default", 4: "3"},
            {0: "5-bold", 1: "bold", 2: "6-bold", 3: "bold", 4: "3-bold"},
            always_in_focus=True,
            column_sizing=[
                urwid.FIXED,
                urwid.FIXED,
                urwid.FIXED,
                urwid.WEIGHT,
                urwid.FIXED,
            ],
            column_align=[
                urwid.LEFT,
                urwid.RIGHT,
                urwid.RIGHT,
                urwid.RIGHT,
                urwid.RIGHT,
            ],
            headers=["UID", "Entries", "Size", "Path", "Last Modified"],
        )
        self.headers = urwid.Columns(
            [("fixed", 1, urwid.Text(" "))]
            + [
                (sizing, width, bold_text(text, align=align))
                for sizing, width, text, align in zip(
                    self.table._column_sizing,
                    self.table._column_widths.values(),
                    self.table._headers,
                    self.table._column_align,
                )
            ],
            dividechars=self.table.items[0].DIVIDECHARS,
        )
        self.header = urwid.Pile([urwid.Divider(), self.headers, urwid.Divider()])
        self.footer = urwid.Filler(
            urwid.Padding(cEdit("Search: ", "", align="left"), left=3), top=1, bottom=1
        )
        urwid.connect_signal(self.footer.base_widget, "change", self.on_search_change)

        # Load the sizes of the databases in a separate thread to decrease loading time
        def get_sizes():
            for key in data:
                self.data[key]["size"] = (
                    subprocess.check_output(["du", "-sh", data[key]])
                    .split()[0]
                    .decode("utf-8")
                )
            for item in self.table.body:
                item.content[2] = str(self.data[item.content[0]]["size"])
                item._invalidate()
            Caller.main_loop.draw_screen()

        threading.Thread(target=get_sizes).start()

    def _process_data(self, data: dict, mtime: dict) -> dict:
        """
        Process the given data and mtime dictionaries to create a new
        dictionary with additional information.

        Parameters:
            - data (dict): A dictionary containing keys as paths and values as
              corresponding data.
            - mtime (dict): A dictionary containing keys as paths and values as
              corresponding modification times.

        Returns:
            - dict: A new dictionary with keys from mtime sorted in descending
              order of modification time. Each key in the new dictionary
              contains a sub-dictionary with the following keys:
                - "path": The original path from the data dictionary.
                - "mtime": The modification time from the mtime dictionary.
                - "entries": The number of subdirectories in the path.
                - "size": The size of the path in a human-readable format.

        Raises:
            - subprocess.CalledProcessError: If an error occurs while running
              the 'du' command to get the size of the path.
        """
        new_data = {
            key: {
                "path": data[key],
                "mtime": mtime[key],
                "entries": len(
                    [
                        i
                        for i in os.listdir(data[key])
                        if os.path.isdir(os.path.join(data[key], i))
                    ]
                ),
                "size": "...",
            }
            for key in sorted(mtime, key=mtime.get, reverse=True)
        }
        return new_data

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        """Process keypress events for the current widget.

        Args:
            size (tuple[int, int]): The size of the widget.
            key (str): The key that was pressed.

        Returns:
            str | None: The result of processing the keypress event.
        """
        if key in ("ctrl n", "ctrl p", "up", "down"):
            return self.table.keypress(size, key)
        if key == "enter":
            # check if something is selected
            if not self.table.focus:
                return
            # get UID of the selected action item
            item = self.table.focus.base_widget.content[0]

            def load_data():
                with Spinner(caption=f"[ID: {item}] Reading data ") as spinner:
                    loading_widget = urwid.Overlay(
                        cRoundedLineBox(spinner, focus_map="default"),
                        Caller.main_loop.widget,
                        "center",
                        ("relative", 50),
                        "middle",
                        ("pack"),
                    )
                    Caller.enter_widget(loading_widget)

                    # Create databaase widget
                    database = Database(item, bb.Manager(uid=item))
                    # Exit the loading widget
                    Caller.exit_widget()

                Caller.enter_widget(database)

            threading.Thread(target=load_data).start()

            return
        return super().keypress(size, key)

    def on_search_change(self, edit, new_edit_text):
        """Updates the table body based on a fuzzy search of the new edit
        text.

        Arguments:
            - edit (str): The current edit text in the search bar.
            - new_edit_text (str): The new edit text entered by the user.

        Returns:
            - None
        """
        filtered = self.fuzzy_search(new_edit_text)
        self.table.body[:] = filtered
        if filtered:
            self.table.focus_position = 0

    def fuzzy_search(self, query):
        """Perform a fuzzy search on the data based on the given query.

        Args:
            - query (str): The query string to search for.

        Returns:
            - list: A list of items that match the fuzzy search criteria.

        Raises:
            - None

        This method takes a query string and performs a fuzzy search on the
        data stored in the object. If the query string is empty, it will return
        all items in the data. Otherwise, it will return items that have a
        partial ratio match of greater than 70 with the query string.
        """
        # item: list[uid, entries, size, path, mtime]
        if query == "":
            return [item for item in self.table.items]

        res = [
            item
            for item in self.table.items
            if fuzz.partial_ratio(item.content[3], query) > 70
        ]
        return res

    def _create_item(self, uid) -> list:
        item = [
            uid,
            self.data[uid]["entries"],
            self.data[uid]["size"],
            self.data[uid]["path"],
            datetime.fromtimestamp(self.data[uid]["mtime"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        ]
        return cActionItem([str(i) for i in item])


class IndexRemote(Index):
    def __init__(self, *args, remote: Remote) -> None:
        self.remote = remote
        data = remote.read_table().set_index("id").to_dict()["path"]
        super().__init__(data)

    def _initialize(self, data: dict) -> None:
        self.data = data

        def bold_text(text, *args, **kwargs):
            return urwid.AttrWrap(urwid.Text(text, *args, **kwargs), "bold")

        self.table = cListBoxSelectionCharacter(
            [self._create_item(key) for key in self.data],
            {0: "5", 1: "default"},
            {0: "5-bold", 1: "bold"},
            always_in_focus=True,
            column_sizing=[urwid.FIXED, urwid.WEIGHT],
            column_align=[urwid.LEFT, urwid.RIGHT],
            headers=["UID", "Path"],
        )
        self.headers = urwid.Columns(
            [("fixed", 1, urwid.Text(" "))]
            + [
                (sizing, width, bold_text(text, align=align))
                for sizing, width, text, align in zip(
                    self.table._column_sizing,
                    self.table._column_widths.values(),
                    self.table._headers,
                    self.table._column_align,
                )
            ],
            dividechars=self.table.items[0].DIVIDECHARS,
        )
        self.header = urwid.Pile([urwid.Divider(), self.headers, urwid.Divider()])
        self.footer = urwid.Filler(
            urwid.Padding(cEdit("Search: ", "", align="left"), left=3), top=1, bottom=1
        )
        urwid.connect_signal(self.footer.base_widget, "change", self.on_search_change)

    def _create_item(self, uid) -> list:
        item = [
            uid,
            self.data[uid],
        ]
        return cActionItem([str(i) for i in item])

    def keypress(self, size: tuple, key: str) -> Union[str, None]:
        if key == "enter":
            # check if something is selected
            if not self.table.focus:
                return
            # get UID of the selected action item
            item = self.table.focus.base_widget.content[0]

            def load_data():
                with Spinner(caption=f"[ID: {item}] Reading data ") as spinner:
                    loading_widget = urwid.Overlay(
                        cRoundedLineBox(spinner, focus_map="default"),
                        Caller.main_loop.widget,
                        "center",
                        ("relative", 50),
                        "middle",
                        ("pack"),
                    )
                    Caller.enter_widget(loading_widget)

                    # Create databaase widget
                    database = Database(
                        item, self.remote[item], _remote_name=self.remote.remote_name
                    )
                    # Exit the loading widget
                    Caller.exit_widget()

                Caller.enter_widget(database)

            threading.Thread(target=load_data).start()

            return

        if key == "ctrl r":

            def read_process_output(widget):
                with Spinner(caption="Downloading index file ") as spinner:
                    Caller.widget_stack[-1]._frame.footer.contents[1] = (
                        spinner,
                        ("pack", None),
                    )
                    # download the index file
                    with self.remote.fetch_index() as process:
                        for line in process.stdout:
                            widget.set_text(widget.text + line)
                            Caller.main_loop.draw_screen()
                    Caller.exit_widget()
                    Caller.main_loop.draw_screen()

            text_widget = urwid.Text("")
            popup = cPopup(
                urwid.Filler(text_widget, valign=urwid.TOP),
                title="Downloading index file",
                footer="",
            )
            Caller.enter_widget(popup)
            Caller.main_loop.draw_screen()
            threading.Thread(target=read_process_output, args=(text_widget,)).start()
            return

        return super().keypress(size, key)

    def fuzzy_search(self, query):
        # item: list[uid, path]
        if query == "":
            return [item for item in self.table.items]

        res = [
            item
            for item in self.table.items
            if fuzz.partial_ratio(item.content[1], query) > 70
        ]
        return res
