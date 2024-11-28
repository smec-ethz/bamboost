from __future__ import annotations

from typing import Any, Iterable

import urwid

from ..common import VERTICAL_KEYS, Caller, palette
from . import custom_widgets as cw


class AutocompleteEdit(cw.cEdit):
    """Edit widget with autocomplete suggestions. Need custom container to display popup,
    which controls the rendering of the screen.

    Args:
        caption (str): The caption to display (e.g. ":")
        edit_text (str): The initial text
        ...
    """

    ATTRS = ("0-reverse", "4-reverse-bold")

    def __init__(self, suggestions: Iterable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_suggestions = suggestions
        self.suggestions = self.all_suggestions
        self.popup = cw.cListBox(urwid.SimpleFocusListWalker([]), wrap=False)
        self.show_popup = False

    def keypress(self, size, key):
        if key == "tab":
            # Use Tab key to select suggestions
            if not self.show_popup:
                return None
            self.set_edit_text(self.popup.get_focus()[0].base_widget.text)
            self.set_edit_pos(len(self.edit_text))
            self.show_popup = False
        elif key == "<0>":
            # Ctrl + Space to toggle popup
            self.show_popup = not self.show_popup
            if self.show_popup:
                self._update_popup(force_show=True)
        elif key in VERTICAL_KEYS and self.show_popup:
            self.popup.keypress((size[0], 1), key)
        else:
            super().keypress(size, key)
            self._update_popup()

    def _update_popup(self, force_show: bool = False) -> None:
        # Logic to update and display popup with suggestions
        if not force_show:
            self.show_popup = self.edit_text != ""
        self.suggestions = {s for s in self.suggestions if s.startswith(self.edit_text)}
        if not self.suggestions:
            self.show_popup = False
        self.popup.body = urwid.SimpleFocusListWalker(
            [
                urwid.AttrMap(urwid.Padding(urwid.Text(s), left=1), *self.ATTRS)
                for s in self.suggestions
            ]
        )


class AutocompleteContainer(urwid.WidgetWrap):
    """
    Custom container to hold the auto complete edit widget. Make sure the edit widget is
    a part of the container (e.g. in the footer of a frame)

    Args:
        edit (AutocompleteEdit): The edit widget with autocomplete suggestions
        container (urwid.Widget): The main container to be rendered
    """

    def __init__(
        self, edit: AutocompleteEdit, container: urwid.Widget, *args, **kwargs
    ):
        self.edit = edit
        self.container = container
        super().__init__(self.container)

    def render(self, size, focus=False) -> urwid.CompositeCanvas:
        # Render the main frame
        canvas = super().render(size, focus)
        canvas = urwid.CompositeCanvas(canvas)
        canvas.set_depends([self.edit, self.edit.popup])

        if not self.edit.show_popup:
            return canvas

        # Get the number of entries in the popup
        entries = len(self.edit.popup.body)
        if entries == 0:
            return canvas

        # Size and position the popup above the cursor
        # x, _ = self.edit.position_coords(size[0] - 20, self.edit.edit_pos)
        if not self.get_cursor_coords(size):
            return canvas
        x, y = self.get_cursor_coords(size)
        height = min(entries, 20, y)
        width = max(len(s) for s in self.edit.suggestions) + 2

        # Overlay the popup on the main frame
        canvas.overlay(
            self.edit.popup.render((width, height), focus=focus),
            x,
            y - height,
        )

        return canvas


# Application entry point
def main():
    Caller.main_loop = urwid.MainLoop(urwid.SolidFill(" "), palette)
    suggestions = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "apple pie",
        "android",
    ]
    edit = AutocompleteEdit(suggestions, ":")
    frame = AutocompleteContainer(edit, urwid.Filler(edit))
    Caller.main_loop.widget = frame
    Caller.main_loop.run()


if __name__ == "__main__":
    main()
