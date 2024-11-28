from __future__ import annotations

import os

import h5py
import numpy as np
import urwid
from rich.traceback import install

from bamboost.tui.common import Caller, palette
from bamboost.tui.keybinds import cKeybindsOverlay
from bamboost.tui.parser.parser import ArgumentParser
from bamboost.tui.widgets import custom_widgets as cw
from bamboost.tui.widgets.autocomplete_edit import AutocompleteContainer
from bamboost.tui.widgets.command_line import CommandLine
from bamboost.tui.widgets.custom_widgets import SelectionIcon
from bamboost.tui.widgets.rounded_line_box import cRoundedLineBox
from bamboost.tui.keybinds import Keybind

install()

FOOTER = "q: Exit widget | Q: Quit app | ?: Keybindings | : Command | tab: Toggle focus"


class FocusedList(urwid.ListBox):
    """List box with a focused item even if the list is not in focus."""

    def render(self, size, focus=False):
        return super().render(size, focus=True)


class AttrsList(cw.cListBoxSelectionCharacter):
    def __init__(self, parent: HDFView, attr_map: dict = None, focus_map: dict = None):
        self.parent = parent
        self.attr_map = attr_map
        self.focus_map = focus_map

    def update(self, key: str = ""):
        super().__init__(self._get_action_items(key), self.attr_map, self.focus_map)
        return self

    def _get_action_items(self, key: str = ""):
        return [self.ActionItemPatched(i) for i in self.parent.get_attrs(key)]

    class ActionItemPatched(cw.cActionItem):
        def rows(self, size, focus=False):
            wide = size[0] > self.column_widths[0] * 1.8
            self.widget = urwid.AttrMap(
                urwid.Columns(
                    [("fixed", 1, SelectionIcon())]
                    + [
                        self._render_column(i, text, focus, wide)
                        for i, text in enumerate(self.content)
                    ],
                    dividechars=1,
                ),
                self.column_attr_map,
                self.column_focus_map,
            )
            return self.widget.rows(size, focus)

        def render(self, size, focus=False):
            return self.widget.render(size, focus)

        def _render_column(
            self, i: int, text: str, focus: bool = False, wide: bool = False
        ):
            if i == 0 and self.column_widths and wide:
                return (
                    "fixed",
                    self.column_widths[i],
                    urwid.AttrMap(urwid.Text(text), i),
                )

            return ("weight", 1, urwid.AttrMap(urwid.Text(text), i))


class HDFView(AutocompleteContainer):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._file_path = os.path.abspath(filename)
        self._current_dir = []
        self._focus_stack = []

        self.current_dir_list = FocusedList(
            urwid.SimpleFocusListWalker(self.get_items())
        )
        self.parent_dir_list = FocusedList(urwid.SimpleFocusListWalker([]))
        self.preview = urwid.ListBox(urwid.SimpleListWalker([]))

        self.header = urwid.Pile(
            [urwid.AttrWrap(urwid.Text(self._file_path), "bold"), urwid.Divider()]
        )
        self.placehold_footer = urwid.AttrWrap(urwid.Text(FOOTER), "footer")
        self.command_line = CommandLine(ArgumentParser())

        self.navigator = urwid.Columns(
            [("weight", 1, self.current_dir_list), ("weight", 2, self.preview)],
            dividechars=2,
            focus_column=0,
        )
        self.attrs_focused = AttrsList(
            self, {0: "5", 1: "8"}, {0: "5-bold", 1: "8-bold"}
        ).update(self.current_dir_list.focus.base_widget.get_text()[0])
        self.attrs_current_dir = AttrsList(
            self, {0: "5", 1: "8"}, {0: "5-bold", 1: "8-bold"}
        ).update()

        self.body = urwid.Pile(
            [
                ("weight", 2, self.navigator),
                (
                    "weight",
                    1.3,
                    urwid.Columns(
                        [
                            cRoundedLineBox(
                                self.attrs_current_dir,
                                title=f"Attributes of current directory",
                                title_align="left",
                                focus_map="5",
                            ),
                            cRoundedLineBox(
                                self.attrs_focused,
                                title=f"Attributes of selected item",
                                title_align="left",
                                focus_map="5",
                            ),
                        ]
                    ),
                ),
            ],
        )
        self.frame = urwid.Frame(
            self.body,
            header=self.header,
            footer=self.placehold_footer,
            focus_part="body",
        )
        super().__init__(self.command_line, self.frame)

        # Add the keybindings
        self.keybinds: dict = {
            "j": (self.navigate_down, "Move focus down", ["ctrl n", "down"]),
            "k": (self.navigate_up, "Move focus up", ["ctrl p", "up"]),
            "l": (self.navigate_enter, "Enter directory", ["enter"]),
            "h": (self.navigate_exit, "Exit directory", ["backspace"]),
            "q": (Caller.exit_widget, "Exit", []),
            ":": (self.enter_command, "Enter command", []),
            "?": (self.show_help, "Show keybindings", []),
            "tab": (self.move_focus, "Move focus between widgets [forwards]", []),
            "shift tab": (
                lambda: self.move_focus(backwards=True),
                "Move focus between widgets [backwards]",
                [],
            ),
            "d": (self.delete_focused, "Delete focused item", []),
        }
        self._keybinds = {key: func for key, (func, _, _) in self.keybinds.items()}
        self._keybinds.update(
            {
                alias: func
                for func, _, aliases in self.keybinds.values()
                for alias in aliases
            }
        )

        urwid.connect_signal(self.command_line, "release_focus", self.execute_command)

    def open_file(self, mode: str = "r"):
        return h5py.File(self._file_path, mode)

    @property
    def current_dir(self):
        if not self._current_dir:
            return "/"
        return "/" + "/".join(self._current_dir)

    @property
    def parent_dir(self):
        if len(self._current_dir) <= 1:
            return "/"
        return "/" + "/".join(self._current_dir[:-1])

    def get_attrs(self, key: str):
        with self.open_file("r") as f:
            attrs = f[f"{self.current_dir}/{key}"].attrs
            if not attrs:
                return ["No attributes"]
            return [[str(i), str(j)] for i, j in attrs.items()]

    def get_items(self, dir: str = None):
        if dir is None:
            dir = self.current_dir

        with self.open_file("r") as f:
            grp = f[dir]
            if isinstance(grp, h5py.Group):
                groups = (key for key in grp.keys() if isinstance(grp[key], h5py.Group))
                datasets = (
                    key for key in grp.keys() if isinstance(grp[key], h5py.Dataset)
                )
                g = [
                    urwid.AttrWrap(urwid.Text(key), "5", "5-reverse") for key in groups
                ]
                d = [
                    urwid.AttrWrap(urwid.Text(key), "6", "6-reverse")
                    for key in datasets
                ]
                return g + d
            elif isinstance(grp, h5py.Dataset):
                return []
            else:
                return []

    def set_preview(self, key: str = None):
        if key is None:
            key = self.current_dir_list.focus.base_widget.get_text()[0]

        with self.open_file("r") as f:
            obj = f[f"{self.current_dir}/{key}"]
            if isinstance(obj, h5py.Dataset):
                if obj.dtype == "O":
                    # If the dataset is of type object, try displaying the string
                    # representation of the object
                    preview_data = urwid.Text(obj[()].decode("utf-8"))
                else:
                    # Assume it is a numeric dataset and display the first 20 rows
                    try:
                        preview_data = urwid.Columns(
                            [
                                ("fixed", 9, urwid.Text("Preview:")),
                                urwid.Pile(
                                    [
                                        *[
                                            (
                                                urwid.Columns(
                                                    [
                                                        (
                                                            "fixed",
                                                            7,
                                                            urwid.Text(f"{i:.3f}"),
                                                        )
                                                        for i in row
                                                    ]
                                                )
                                                if isinstance(row, np.ndarray)
                                                else urwid.Text(f"{row:.3f}")
                                            )
                                            for row in obj[:20]
                                        ],
                                        (
                                            urwid.Text("...")
                                            if obj.shape[0] > 20
                                            else urwid.Text("")
                                        ),
                                    ]
                                ),
                            ]
                        )
                    except:
                        preview_data = urwid.Text(
                            str(obj[:20]) + ("..." if obj.shape[0] > 20 else "")
                        )

                # Construct the preview list with the items
                self.preview.body = urwid.SimpleListWalker(
                    [
                        urwid.Text(
                            ("6-bold", f"{self.current_dir}/{key}"), wrap="clip"
                        ),
                        urwid.Divider(),
                        urwid.Text(
                            ["Shape:   ", ("bold", f"{obj.shape}")],
                            wrap="clip",
                        ),
                        urwid.Text(
                            ["dtype:   ", ("bold", f"{obj.dtype}")],
                            wrap="clip",
                        ),
                        urwid.Divider("\u2500"),
                        preview_data,
                    ]
                )
            elif isinstance(obj, h5py.Group):
                children = self.get_items(f"{self.current_dir}/{key}")
                self.preview.body = urwid.SimpleListWalker(
                    [
                        urwid.Text(
                            ("5-bold", f"{self.current_dir}/{key}"), wrap="clip"
                        ),
                        urwid.Divider(),
                        urwid.Pile(children) if children else urwid.Text(""),
                    ]
                )
            else:
                self.preview.body = urwid.SimpleListWalker([])

    def navigate_down(self):
        try:
            self.current_dir_list.set_focus(self.current_dir_list.focus_position + 1)
            self.attrs_focused.update(
                self.current_dir_list.focus.base_widget.get_text()[0]
            )
            self.set_preview()
        except IndexError:
            return

    def navigate_up(self):
        try:
            self.current_dir_list.set_focus(self.current_dir_list.focus_position - 1)
            self.attrs_focused.update(
                self.current_dir_list.focus.base_widget.get_text()[0]
            )
            self.set_preview()
        except IndexError:
            return

    def navigate_enter(self):
        # If the group is empty, do nothing
        # Or if the current item is a dataset, do nothing
        with self.open_file("r") as f:
            obj = f[
                f"{self.current_dir}/{self.current_dir_list.focus.base_widget.get_text()[0]}"
            ]
            if isinstance(obj, h5py.Dataset) or not obj.keys():
                return

        self._current_dir.append(self.current_dir_list.focus.base_widget.get_text()[0])
        self._focus_stack.append(self.current_dir_list.focus_position)

        # Update the current directory list and the attributes list to reflect the new directory
        self.current_dir_list.body = urwid.SimpleFocusListWalker(self.get_items())
        self.current_dir_list.set_focus(0)
        self.attrs_focused.update(self.current_dir_list.focus.base_widget.get_text()[0])

        # Add parent dir list to navigator contents, if it doesn't exist
        if len(self.navigator.contents) == 2:
            self.navigator.contents.insert(
                0, (self.parent_dir_list, ("weight", 0.75, False))
            )

        # Update the parent directory list and set the focus to the last focused item
        self.parent_dir_list.body = urwid.SimpleFocusListWalker(
            self.get_items(self.parent_dir)
        )
        self.parent_dir_list.set_focus(self._focus_stack[-1])
        self.set_preview()

        # Set the attributes of the current directory
        self.attrs_current_dir.update()

    def navigate_exit(self):
        # If the current directory is the root, do nothing
        if not self._current_dir:
            return

        self._current_dir.pop()
        self.current_dir_list.body = urwid.SimpleFocusListWalker(self.get_items())
        self.current_dir_list.set_focus(self._focus_stack.pop())

        if not self._focus_stack:
            self.parent_dir_list.body = urwid.SimpleFocusListWalker([])
            self.navigator.contents.pop(0)  # remove parent dir list
        else:
            self.parent_dir_list.body = urwid.SimpleFocusListWalker(
                self.get_items(self.parent_dir)
            )
            self.parent_dir_list.set_focus(self._focus_stack[-1])

        self.attrs_focused.update(self.current_dir_list.focus.base_widget.get_text()[0])
        self.set_preview()

        # Set the attributes of the current directory
        self.attrs_current_dir.update()

    def show_help(self):
        if not hasattr(self, "_help_widget"):
            self._help_widget = cKeybindsOverlay(Caller.main_loop, self.keybinds)
        self._help_widget.toggle()

    def move_focus(self, backwards: bool = False):
        # if self.container.focus.focus == self.navigator:
        pile = self.container.focus
        if pile.focus_position == 0:
            pile.focus_position = 1
            pile.focus.focus_position = 0 if not backwards else 1
            return
        if pile.focus_position == 1:
            if not backwards:
                if pile.focus.focus_position == 0:
                    pile.focus.focus_position = 1
                else:
                    pile.focus_position = 0
            else:
                if pile.focus.focus_position == 1:
                    pile.focus.focus_position = 0
                else:
                    pile.focus_position = 0
        return

    def delete_focused(self):
        """
        Delete the focused item
        """
        key = self.current_dir_list.focus.base_widget.get_text()[0]

        def _delete() -> None:
            with self.open_file("a") as f:
                del f[f"{self.current_dir}/{key}"]

            _fp = self.current_dir_list.focus_position
            self.current_dir_list.body = urwid.SimpleFocusListWalker(self.get_items())

            try:
                self.current_dir_list.set_focus(_fp - 1)
            except IndexError:
                pass  # leave at default position (0)

            if self.current_dir_list.body:
                self.attrs_focused.update(
                    self.current_dir_list.focus.base_widget.get_text()[0]
                )
                self.set_preview()

        ui = cw.cConfirmDialog(
            Caller.main_loop,
            f"Are you sure you want to delete {self.current_dir}/{key}?",
            callback=_delete,
        )
        Caller.enter_widget(ui)

    def enter_command(self):
        self.container.footer = self.command_line
        self.command_line.edit_text = ""
        self.container.set_focus("footer")

    def execute_command(self, command: str = None) -> None:
        if command == "q":
            Caller.exit_widget()
        else:
            self.container.footer = self.placehold_footer
            self.container.set_focus("body")

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        if self.container.focus_part == "footer":
            return self.container.footer.keypress(size, key)

        if (
            key not in {"tab", "shift tab", "?", "q"}
            and self.container.focus.focus_position != 0
        ):
            if self.container.focus.focus.focus_position == 0:
                self.attrs_current_dir.keypress((1, 1), key)
            else:
                self.attrs_focused.keypress((1, 1), key)
            return
        if key in self._keybinds and self.container.focus_part == "body":
            self._keybinds[key]()
        else:
            return self.container.keypress(size, key)


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: hdfview <filename>")
        sys.exit(1)
    filename = sys.argv[1]

    main_widget = HDFView(filename)
    Caller.main_loop = urwid.MainLoop(urwid.Pile([]), palette)
    Caller.main_loop.widget = main_widget
    Caller.main_loop.screen.set_terminal_properties(colors=256)
    Caller.widget_stack.append(main_widget)
    Caller.main_loop.run()


if __name__ == "__main__":
    main()
