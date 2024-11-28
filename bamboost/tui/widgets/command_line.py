from __future__ import annotations

import re
from typing import Iterable

import urwid
from bamboost.tui.parser.parser import ArgumentParser

from ..common import Caller
from .autocomplete_edit import VERTICAL_KEYS, AutocompleteEdit
from .custom_widgets import cEditHistory


class CommandLine(AutocompleteEdit, cEditHistory):
    """
    Command line widget. Inherits from AutocompleteEdit and cEditHistory to
    enable completion and history.

    Args:
        parser: The parser to use for the command line

    Attributes:
        options: some options for the command line
        parser: The parser to use for the command line
        suggestions: The current suggestions for the command line
    """

    _selectable = True
    signals = ["release_focus", "execute_command"]

    def __init__(
        self,
        parser: ArgumentParser,
        *args,
        caption: str = ":",
        prefix: str = "",
        **kwargs,
    ):
        self.options = {"caption": caption, "prefix": prefix}
        self.parser = parser
        # self.suggestions = self.parser.functions
        self.suggestions = []
        super().__init__(self.suggestions, caption, *args, **kwargs)

    @property
    def suggestions(self) -> Iterable[str]:
        return self._suggestions

    @suggestions.setter
    def suggestions(self, suggestions: Iterable[str]) -> None:
        self._suggestions = suggestions
        try:
            self._update_popup()
        except AttributeError:
            pass

    def _reset(self) -> None:
        """
        Reset the command line
        """
        self.parser._reset()
        self.set_edit_text(self.options["prefix"] + "")

    def keypress(self, size, key):
        if key == "enter":
            if self.edit_text == "q":
                Caller.exit_widget()
                return None

            # Append the command to the history and emit the release_focus signal
            self.history.append(self.edit_text)
            urwid.emit_signal(self, "execute_command", self.edit_text)

        elif key == "esc":
            urwid.emit_signal(self, "release_focus")

        # elif key == "backspace" and self.edit_text == "":
        #     urwid.emit_signal(self, "release_focus")

        elif key == "tab":  # accept the suggestion
            if not self.show_popup:
                return None
            new_text = self.popup.get_focus()[0].base_widget.text.strip()
            self.edit_text = (
                " ".join(re.split(r"\s+", self.edit_text)[:-1] + [new_text]) + " "
            )
            self.edit_pos = len(self.edit_text)
            self.suggestions = self.parser.current_suggestions(self.edit_text)

        elif key == "<0>":  # Ctrl + Space to toggle popup
            self.show_popup = not self.show_popup
            if self.show_popup:
                self.suggestions = self.parser.current_suggestions(self.edit_text)

        elif key not in VERTICAL_KEYS | {"<0>"}:  # type letter and update suggestions
            super().keypress(size, key)
            self.suggestions = self.parser.current_suggestions(self.edit_text)
        else:
            super().keypress(size, key)

    def _update_popup(self, *args, **kwargs) -> None:
        # Logic to update and display popup with suggestions
        if not self.show_popup and self.edit_text != "":
            self.show_popup = True
        if not self.suggestions:
            self.show_popup = False
        self.popup.body = urwid.SimpleFocusListWalker(
            [
                urwid.AttrMap(urwid.Padding(urwid.Text(s), left=1), *self.ATTRS)
                for s in self.suggestions
            ]
        )
