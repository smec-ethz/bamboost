from __future__ import annotations

from enum import Enum
from itertools import chain
from numbers import Number
from typing import Any, Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd
import urwid
from bamboost.tui.common import Config, palette
from bamboost.tui.keybinds import Keybindings, apply_user_keybinds
from bamboost.tui.widgets.custom_widgets import cListBox

import bamboost.core as bb

DIVIDE_CHARS = 2
SORT_CHARACTERS = {
    True: "\u25b2",
    False: "\u25bc",
}
SELECT_CHARACTER = "❯"


class Part(Enum):
    # Enum for the focus part of the row widget
    # the value is the focus_col of the Columns widget
    PINNED = 1
    SCROLLABLE = 3


class SortOrder(Enum):
    ASCENDING = True
    DESCENDING = False


class ColumnsAlwaysFocused(urwid.Columns):
    """A Columns widget that always renders with focus."""

    def render(self, size, focus: bool = False):
        return super().render(size, focus=True)


class SelectionIcon(urwid.Text):
    # Icon to indicate the selected row
    # focus_cb is a callback to check if the row the icon belongs to is focused
    def __init__(self, focus_cb: callable):
        self.icon = SELECT_CHARACTER
        self.focus_cb = focus_cb
        super().__init__(self.icon, align="center")

    def render(self, size, focus=False):
        if self.focus_cb():
            self.set_text(("4", self.icon))
        else:
            self.set_text(" ")
        return super().render(size, focus)


class ColumnFocusIndicator(urwid.Text):
    _icons = {
        SortOrder.ASCENDING: "\u25b2",
        SortOrder.DESCENDING: "\u25bc",
    }

    def __init__(self, table: Table):
        self.table = table
        super().__init__("a", align="center")

    def render(self, size, focus=False):
        if focus:
            self.set_text(("4", self._icons[self.table._sort_order]))
        else:
            self.set_text(" ")
        return super().render(size, focus)


class Table(urwid.Frame):
    """Table widget for displaying a pandas DataFrame. The table is displayed as a
    urwid.Frame with the Header being the column names and the body being the data.
    Each row is a Row widget is derived from urwid.Columns and contains a list of
    cells. Each cell is a urwid.Text widget.

    AttrMap is used for styling the table.


    """

    signals = [
        "enter_file",
        "set_footer",
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        # Store a copy of the dataframe to apply filters and sorting
        self._filtered_df = df

        self._column_sizes = self._get_column_sizes(df)
        self._sort_column = None
        self._focus_column = "time_stamp"
        self._sort_order = SortOrder.ASCENDING

        self.columns = np.array(
            df.columns.tolist()
        )  # columns ordered as in the DataFrame
        self.pinned_columns = ["id"]
        self.scrollable_columns = [
            col for col in df.columns if col not in self.pinned_columns
        ]

        # fmt: off
        self.keybinds = (
            Keybindings(self, group="table")
            .new("right", "l", Table.navigate_right, "Navigate right", ["right", "ctrl f"])
            .new("left", "h", Table.navigate_left, "Navigate left", ["left", "ctrl b"])
            .new("sort", "s", lambda *_: self.sort(self._get_column_name_in_focus()), "Sort column in focus")
            .new("reset", "r", Table.reset, "Reset table")
            .new("enter", "enter", Table.enter, "Enter hdfview")
        ) 
        # fmt: on

        # Shelf of all rows
        self._rows: dict[str, RowEntry] = {
            record["id"]: RowEntry(self, record)
            for record in df.to_records(index=False)
        }
        self.entries = cListBox(
            urwid.SimpleFocusListWalker(self._id_list_to_rows(df["id"].values))
        )
        self.headers = RowHeader(self, df.columns)

        apply_user_keybinds(self.keybinds, "table")
        # merge keybinds from self.entries and assign to group table
        self.keybinds.merge(self.entries.keybinds, group="table")
        self.keybinds.resolve_mapping()

        super().__init__(
            self.entries,
            header=urwid.Pile([self.headers, urwid.Divider("\u2500")]),
            focus_part="body",
        )

    def _id_list_to_rows(self, id_list: list[str]) -> list[RowEntry]:
        """Returns a list of rows for the given list of ids."""
        return [self._rows[id] for id in id_list]

    def _invalidate(self) -> None:
        self._column_sizes = self._get_column_sizes(self._df)
        self.entries._invalidate()
        return super()._invalidate()

    def render(self, size, focus: bool = False):
        # Always render the table with focus
        return super().render(size, focus=True)

    @property
    def pinned_columns(self) -> np.ndarray:
        return self.columns[np.isin(self.columns, list(self._pinned_columns))]

    @pinned_columns.setter
    def pinned_columns(self, columns: Set[str]) -> None:
        self._pinned_columns = set(columns)
        if not hasattr(self, "entries"):
            return

        for row in chain(self.entries.body, [self.headers]):
            row._invalidate()

    @property
    def scrollable_columns(self) -> np.ndarray:
        return self.columns[np.isin(self.columns, list(self._scrollable_columns))]

    @scrollable_columns.setter
    def scrollable_columns(self, columns: Set[str]) -> None:
        self._scrollable_columns = set(columns)
        if not hasattr(self, "entries"):
            return

        for row in chain(self.entries.body, [self.headers]):
            row._invalidate()

    def loc(self, id: str) -> RowEntry:
        return self._rows[id]

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        if key in self.keybinds.keys():
            self.keybinds.call(key, size=size)
            return
        self.keybinds.reset_submap()
        return super().keypress(size, key)

    def _get_column_sizes(self, df: pd.DataFrame, max_column_width: int = 30) -> dict:
        """Calculate the maximum width of each column in the DataFrame. This is used to set
        the width of each cell in the table. The maximum width is calculated by taking the
        length of the column name and the length of the longest entry in the column.

        Args:
            - df (pd.DataFrame): The DataFrame to calculate the column sizes for.
        """
        column_sizes = dict()
        for col in df.columns:
            max_length = len(col)
            for entry in df[col].values:
                if isinstance(entry, np.ndarray) and (entry.size > 6 or entry.ndim > 1):
                    max_length = max(max_length, len(f"Array {entry.shape}"))
                elif isinstance(entry, np.datetime64):
                    max_length = 19
                else:
                    max_length = max(max_length, len(str(entry)))

            # Ignore max-column-width for certain columns
            if col in Config.table.get("no-cutoff-columns", []):
                column_sizes[col] = max_length
            else:
                column_sizes[col] = min(max_length, max_column_width)
        return column_sizes

    def _get_column_name_in_focus(self) -> str:
        """Returns the name of the column in focus."""
        return self.headers.get_column_name_in_focus()

    def _get_entry_in_focus(self) -> np.record:
        """Returns the column name of the column in focus."""
        return self.entries.focus.record

    def navigate_right(self) -> None:
        for row in chain(self.entries.body, [self.headers]):
            row.navigate_right()

    def navigate_left(self) -> None:
        for row in chain(self.entries.body, [self.headers]):
            row.navigate_left()

    def enter(self) -> None:
        id = self._get_entry_in_focus()["id"]
        urwid.emit_signal(self, "enter_file", id)

    def go_to_column(self, column: str, **kwargs) -> None:
        if column in self._pinned_columns:
            index = np.where(self.pinned_columns == column)[0][0]
            part = Part.PINNED
        elif column in self.scrollable_columns:
            index = np.where(self.scrollable_columns == column)[0][0]
            part = Part.SCROLLABLE
        else:
            raise KeyError(f"Column {column} is not displayed")

        self._focus_column = column
        for row in chain(self.entries.body, [self.headers]):
            row.set_focus(part, int(index))

    def reset_data(self, df: pd.DataFrame) -> None:
        self._df = df
        self._filtered_df = df
        self._rows = {
            record["id"]: RowEntry(self, record)
            for record in df.to_records(index=False)
        }
        self.entries.body[:] = self._id_list_to_rows(df["id"].values)
        self.headers._invalidate()

    # -----------------------------------
    # Functions for sorting and filtering
    # -----------------------------------

    def sort(self, column: str, **kwargs) -> None:
        """Sort the table by the given column."""
        close = kwargs.get("close", None)
        reverse = kwargs.get("reverse", None)

        self.go_to_column(column)

        if close is not None:
            return self.sort_closest(column, **kwargs)

        if reverse is None:
            self._sort_order = SortOrder(not self._sort_order.value)
            reverse = self._sort_order.value

        # Sort the DataFrame by the given column
        self._filtered_df = self._filtered_df.sort_values(column, ascending=reverse)
        # Update the rows in the table
        self.entries.body[:] = self._id_list_to_rows(self._filtered_df["id"].values)
        self.headers._invalidate()

    def sort_closest(self, column: str, **kwargs) -> None:
        """Sort by the closest value in a column"""
        value = kwargs["close"]
        value = float(value) if value.replace(".", "", 1).isdigit() else value
        try:
            self._filtered_df = self._df.iloc[
                (self._df[column] - value).abs().argsort()
            ]
            self.entries.body[:] = self._id_list_to_rows(self._filtered_df["id"].values)
        except KeyError:
            raise KeyError(f"Column {column} does not exist")

    def reset(self, **kwargs) -> None:
        """Reset the table to the original DataFrame."""
        self._filtered_df = self._df
        self.entries.body[:] = self._id_list_to_rows(self._df["id"].values)
        self.go_to_column(self._get_column_name_in_focus())
        self.headers._invalidate()

    def filter(self, column: str, **kwargs) -> None:
        """Filter by a column

        Args:
            column (str): The column to filter by.
            kwargs: these are the arguments parsed by argparse from the string
                input in the command line.
        """
        self.go_to_column(column)

        def filter_op(op: str, val: str, as_string: bool = False) -> pd.DataFrame:
            if not as_string:
                try:
                    val = float(val)
                except ValueError:
                    pass
            if op == "eq":
                return self._filtered_df[self._filtered_df[column] == val]
            elif op == "ne":
                return self._filtered_df[self._filtered_df[column] != val]
            elif op == "gt":
                return self._filtered_df[self._filtered_df[column] > val]
            elif op == "lt":
                return self._filtered_df[self._filtered_df[column] < val]
            elif op == "ge":
                return self._filtered_df[self._filtered_df[column] >= val]
            elif op == "le":
                return self._filtered_df[self._filtered_df[column] <= val]
            elif op == "contains":
                return self._filtered_df[
                    self._filtered_df[column].str.contains(val, case=False)
                ]
            else:
                return self._filtered_df

        as_string = kwargs.pop("as_string", False)
        for opt, value in kwargs.items():
            if value is not None:
                self._filtered_df = filter_op(opt, value, as_string)
                self.entries.body[:] = self._id_list_to_rows(
                    self._filtered_df["id"].values
                )

    def pin_column(self, column: str) -> None:
        """Pin a column to the left."""
        if column in self._pinned_columns:
            self._pinned_columns.remove(column)
            self._scrollable_columns.add(column)
        else:
            self._scrollable_columns.remove(column)
            self._pinned_columns.add(column)
        for row in chain(self.entries.body, [self.headers]):
            row._invalidate()


class Row(urwid.AttrMap):
    _cells: Dict[str, CellEntry]

    pinned_columns: urwid.Columns
    scrollable_columns: urwid.Columns

    _widget: urwid.Columns
    _focus_part: Part
    base_widget: urwid.Columns
    table: Table

    def __init__(self, widget: urwid.Columns, attr_map: dict, focus_map: dict) -> None:
        super().__init__(widget, attr_map, focus_map)
        self.focus_part = Part.SCROLLABLE
        self.base_widget.focus.set_focus(0)

    def _invalidate(self) -> None:
        self._update_sub_column_widgets()
        return super()._invalidate()

    @property
    def focus_part(self) -> Part:
        return self._focus_part

    @focus_part.setter
    def focus_part(self, part: Part) -> None:
        self._focus_part = part
        self.base_widget.focus_col = part.value

    @property
    def focus(self) -> urwid.Columns:
        return self.base_widget.focus

    def set_focus(self, part: Part, focus: int) -> None:
        self.focus_part = part
        self.focus.focus_position = focus

    def edit_cell_text(self, column: str, text: str) -> None:
        self._cells[column][2].base_widget.set_text(text)

    def navigate_right(self) -> None:
        try:
            self.focus.focus_position += 1
        except IndexError:
            if self.focus_part == Part.PINNED:
                self.focus_part = Part.SCROLLABLE
                self.scrollable_columns.focus_position = 0

    def navigate_left(self) -> None:
        try:
            self.focus.focus_position -= 1
        except IndexError:
            if self.focus_part == Part.SCROLLABLE:
                self.focus_part = Part.PINNED
                self.focus.focus_position = len(self.pinned_columns.contents) - 1

    def _col_list_to_cells(self, columns: list[str]) -> list[CellEntry]:
        """Returns a list of cells for the given list of columns."""
        return [self._cells[col] for col in columns]

    def _construct_sub_column_widgets(self) -> None:
        """Constructs the column widgets for the fixed and scrollable columns."""
        self.pinned_columns = urwid.Columns(
            self._col_list_to_cells(self.table.pinned_columns), dividechars=DIVIDE_CHARS
        )
        self.scrollable_columns = urwid.Columns(
            self._col_list_to_cells(self.table.scrollable_columns),
            dividechars=DIVIDE_CHARS,
        )

    def _update_sub_column_widgets(self) -> None:
        """Updates the column widgets for the fixed and scrollable columns.
        This is used to update the column widgets when the columns are pinned or unpinned.
        """
        self.pinned_columns.contents[:] = [
            (c[2], (urwid.widget.GIVEN, c[1], False))
            for c in self._col_list_to_cells(self.table.pinned_columns)
        ]
        self.scrollable_columns.contents[:] = [
            (c[2], (urwid.widget.GIVEN, c[1], False))
            for c in self._col_list_to_cells(self.table.scrollable_columns)
        ]


class RowEntry(Row):
    def __init__(self, table: Table, record: np.record) -> None:
        self.table = table
        self.record = record
        self.columns = record.dtype.names

        # Construct cells for each column (immutable tuples of fixed width and text)
        self._cells = {
            col: CellEntry(col, record[col], self.table._column_sizes[col])
            for col in self.columns
        }
        self._construct_sub_column_widgets()
        self.selection_icon = SelectionIcon(lambda: self.table.entries.focus == self)

        self._widget = urwid.Columns(
            [
                ("fixed", 1, self.selection_icon),
                ("pack", self.pinned_columns),
                ("fixed", 1, urwid.Text("\u2502")),
                self.scrollable_columns,
            ],
            dividechars=1,
        )

        super().__init__(self._widget, *self._get_color_coding(record, self.columns))

    def render(self, size, focus: bool = False):
        self.selection_icon._invalidate()
        return super().render(size, focus)

    def recreate(self) -> None:
        """Reload the record from the DataFrame and update the cells."""
        self.record = self.table._df[
            self.table._df["id"] == self.record["id"]
        ].to_records()[0]
        self._cells = {
            col: CellEntry(col, self.record[col], self.table._column_sizes[col])
            for col in self.columns
        }
        # Update the contents of the column widgets
        self._update_sub_column_widgets()
        # Update the attribute maps for this row
        self.attr_map, self.focus_map = self._get_color_coding(
            self.record, self.columns
        )

    def _get_color_coding(
        self, record: np.record, columns: Iterable[str]
    ) -> Tuple[dict, dict]:
        """Returns the attribute and focus maps for the given record and columns."""
        if Config.theme == "highlight":
            hl = "reverse"
        elif Config.theme == "default":
            hl = "bold"

        attr_map = dict()
        focus_map = dict()
        # Color the cells based on the type of the value
        for col in columns:
            val = record[col]
            # Datetime
            if isinstance(val, np.datetime64):
                attr_map[col] = "7"
                focus_map[col] = f"7-{hl}"
            # None value
            elif val is None:
                attr_map[col] = "9"  # dark grey
                focus_map[col] = f"9-{hl}"
            # Numpy array
            elif isinstance(val, np.ndarray):
                attr_map[col] = "6"
                focus_map[col] = f"6-{hl}"
            # Boolean
            elif isinstance(val, (bool, np.bool_)):
                attr_map[col] = "3" if val else "2"
                focus_map[col] = f"3-{hl}" if val else f"2-{hl}"
            # Status
            elif col == "status":
                if val in {"Running", "Started"}:
                    attr_map[col] = "4"
                    focus_map[col] = f"4-{hl}"
                elif val in {"Finished", "Completed", "Success"}:
                    attr_map[col] = "3"
                    focus_map[col] = f"3-{hl}"
                elif isinstance(val, str) and ("Failed" in val):
                    attr_map[col] = "2"
                    focus_map[col] = f"2-{hl}"
            elif isinstance(val, Number):
                attr_map[col] = "default"
                focus_map[col] = hl
            elif col == "id":
                attr_map[col] = "5"
                focus_map[col] = f"5-{hl}"
            else:
                # default
                attr_map[col] = "default"
                focus_map[col] = hl

        return attr_map, focus_map


class RowHeader(Row):
    def __init__(self, table: Table, columns: Iterable[str]) -> None:
        self.table = table
        self.columns = columns

        self._cells = {
            col: CellHeader(col, self.table._column_sizes[col], table)
            for col in self.columns
        }
        self._construct_sub_column_widgets()

        self._widget = ColumnsAlwaysFocused(
            [
                ("fixed", 1, urwid.Text("\n")),
                ("pack", self.pinned_columns),
                ("fixed", 1, urwid.Text("\n\u2502")),
                self.scrollable_columns,
            ],
            dividechars=1,
        )

        super().__init__(self._widget, None, None)

    def get_column_name_in_focus(self) -> str:
        return self.focus.focus.base_widget.contents[1][0].get_text()[0]

    def render(self, size, focus: bool = False):
        # invalidate the focus indicator
        self.focus.focus.contents[0][0]._invalidate()
        return super().render(size, focus)


class CellHeader(tuple):
    def __new__(cls, column: str, width: int, table: Table) -> CellHeader:
        widget = urwid.Text(column)
        widget = urwid.Pile([ColumnFocusIndicator(table), widget])
        return super().__new__(cls, ("fixed", width, widget))


class CellEntry(tuple):
    def __new__(cls, column: str, value: Any, width: int) -> CellEntry:
        # Cell formatting baased on the type of the value
        if isinstance(value, np.datetime64):
            # format the datetime
            value = pd.to_datetime(value).strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, np.ndarray) and (value.size > 6 or value.ndim > 1):
            value = f"array {value.shape}"

        text = urwid.Text(str(value), wrap="ellipsis")
        text = urwid.AttrMap(text, column, column)
        return super().__new__(cls, ("fixed", width, text))


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from rich import traceback

    traceback.install()

    data = {
        "id": [1, 2, 3, 4, 5],
        "time_stamp": pd.to_datetime(
            ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]
        ),
        "status": ["Running", "Finished", "Failed", "Started", "Completed"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        "array": [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
            np.array([10, 11, 12]),
            np.array([13, 14, 15]),
        ],
        "submitted": [True, False, True, False, True],
    }

    df = pd.DataFrame(data)
    df = bb.Manager.fromUID["D1740D3FB4"].df
    table = Table(df)
    main_loop = urwid.MainLoop(table, palette=palette)
    main_loop.screen.set_terminal_properties(colors=256)
    main_loop.run()
