import os
import re
import subprocess
import threading
from typing import Union

import urwid

from bamboost.core import Simulation
from bamboost.core.extensions.remote_manager import Remote
from bamboost.core.index import index
from bamboost.tui.common import FIGLET, Caller, Spinner
from bamboost.tui.pages.hdfview import HDFView
from bamboost.tui.parser.parser import ArgumentParser
from bamboost.tui.widgets.autocomplete_edit import AutocompleteContainer
from bamboost.tui.widgets.command_line import CommandLine
from bamboost.tui.widgets.custom_widgets import (
    cActionItem,
    cListBoxSelectionCharacter,
    cPopup,
)
from bamboost.tui.widgets.index import Index, IndexRemote
from bamboost.tui.widgets.rounded_line_box import cRoundedLineBox

FOOTER = urwid.AttrWrap(
    urwid.Text(
        "q: Exit widget | Q: Quit app | enter: Enter | ?: Keybindings", align="left"
    ),
    "footer",
)


class WelcomeUI:
    """
    Welcome screen of the app. Displays a figlet logo and a list of options to choose from.
    """

    def __init__(self) -> None:
        self.welcome = urwid.AttrMap(
            urwid.Text(FIGLET, align="center", wrap="clip"), "8"
        )
        self.listbox = cListBoxSelectionCharacter(
            [
                cActionItem(
                    ["Index", "Show all known databases and enter selected"],
                    callback=self._cb_load_index,
                ),
                cActionItem(
                    ["Remote", "Show index of remote databases"],
                    callback=self._cb_load_remote,
                ),
                cActionItem(
                    [
                        "Scan paths",
                        "Scan specified paths in config file for new databases",
                    ],
                    callback=self._cb_scan,
                ),
                cActionItem(
                    ["Change config", "Open the config file in default editor"],
                    callback=self._cb_open_config,
                ),
                cActionItem(["Exit", "Quit the app"], callback=self._cb_exit),
            ],
            attr_map={0: "5", 1: "default"},
            focus_map={0: "5-bold", 1: "bold"},
        )
        self.frame = urwid.Frame(
            urwid.Pile(
                [
                    (
                        "weight",
                        1,
                        urwid.Filler(self.welcome, valign="bottom", bottom=4),
                    ),
                    (
                        "pack",
                        urwid.Filler(
                            urwid.Padding(
                                self.listbox,
                                width=("relative", 1),
                                min_width=sum(self.listbox._column_widths.values()) + 5,
                                align="center",
                            ),
                            valign="bottom",
                            height=len(self.listbox.items),
                        ),
                    ),
                    (
                        "weight",
                        1,
                        urwid.Filler(
                            urwid.Text(
                                (
                                    "9",
                                    (
                                        "developed with ❤ \n"
                                        "2024, Flavio Lorez and contributors\n"
                                        "https://gitlab.com/cmbm-ethz/bamboost"
                                    ),
                                ),
                                align="center",
                            ),
                            valign="middle",
                        ),
                    ),
                ]
            ),
            footer=FOOTER,
        )

    def _set_footer(self, text: Union[str, urwid.Widget]):
        if isinstance(text, str):
            self.frame.footer = urwid.AttrWrap(urwid.Text(text, align="left"), "footer")
        else:
            self.frame.footer = text

    # ------------------
    # Callback functions
    # ------------------
    def _cb_load_index(self, *args):
        def inner():
            with Spinner(align="left", caption="Loading index ") as spinner:
                self._set_footer(spinner)
                ui = Index(
                    index.IndexAPI.ThreadSafe()
                    .read_table()
                    .set_index("id")
                    .to_dict()["path"]
                )
                overlay = urwid.Overlay(
                    cRoundedLineBox(
                        ui, title="Bamboost Index", title_align="left", focus_map="8"
                    ),
                    Caller.main_loop.widget.base_widget,
                    align="center",
                    width=200,
                    valign="middle",
                    height=30,
                )

            self._set_footer(FOOTER)
            Caller.enter_widget(overlay)

        threading.Thread(target=inner).start()

    def _cb_load_remote(self, item: cActionItem):
        def open_remote_index(remote: Union[str, cActionItem]):
            remote_name = remote if isinstance(remote, str) else remote.content[0]

            def inner():
                with Spinner(caption="Loading remote index ") as spinner:
                    self._set_footer(spinner)
                    ui = IndexRemote(remote=Remote(remote_name, skip_update=True))
                    overlay = urwid.Overlay(
                        cRoundedLineBox(
                            ui,
                            title=f"Bamboost Index - {remote_name} | ctrl r: download index",
                            focus_map="5",
                        ),
                        Caller.main_loop.widget.base_widget,
                        align="center",
                        width=200,
                        valign="middle",
                        height=30,
                    )

                self._set_footer(FOOTER)
                Caller.enter_widget(overlay)

            threading.Thread(target=inner).start()

        # edit = urwid.Filler(cEdit(callback=open_remote_index))
        # ui = cPopup(
        #     edit,
        #     title="Enter remote",
        #     align="center",
        #     width=("relative", 80),
        #     valign="middle",
        #     height=1,
        # )
        listbox = cListBoxSelectionCharacter(
            [cActionItem([name], callback=open_remote_index) for name in Remote.list()],
            attr_map={0: "5"},
            focus_map={0: "5-bold"},
        )
        ui = cPopup(
            listbox,
            title="Select remote",
            align="center",
            width=("relative", 80),
            valign="middle",
            height=len(listbox),
        )
        Caller.enter_widget(ui)

    def _cb_scan(self, *args):
        def inner() -> None:
            with Spinner(caption="Scanning paths ") as spinner:
                self._set_footer(spinner)
                index.IndexAPI.ThreadSafe().scan_known_paths()

            self._set_footer("Paths scanned successfully")

        threading.Thread(target=inner).start()

    def _cb_open_config(self, *args):
        editor = os.environ.get("EDITOR", "vim")
        home = os.path.expanduser("~")
        subprocess.run(
            [editor, os.path.join(home, ".config", "bamboost", "config.toml")],
        )
        Caller.main_loop.screen.clear()

    def _cb_direct_access(self, *args):
        ui = DirectAccess()
        Caller.enter_widget(ui)

    def _cb_exit(self, *args):
        raise urwid.ExitMainLoop()


class DirectAccess(AutocompleteContainer):
    class CustomCommandLine(CommandLine):
        def keypress(self, size, key):
            if key == "tab":
                if not self.show_popup:
                    return None
                new_text = self.popup.get_focus()[0].base_widget.text
                self.edit_text = ":".join(
                    re.split(r"\:+", self.edit_text)[:-1] + [new_text]
                )
                if len(self.edit_text.split(":")) <= 1:
                    self.edit_text += ":"
                self.edit_pos = len(self.edit_text)
                self.suggestions = self.parser.current_suggestions(self.edit_text)
                return

            if key == "meta backspace":
                # Logic to delete the word before the cursor until : or start of the line
                new_pos = (
                    self.edit_text.rfind(":", 0, self.edit_pos - 1) + 1
                    if self.edit_pos > 0
                    else 0
                )
                new_text = self.edit_text[:new_pos] + self.edit_text[self.edit_pos :]
                self.set_edit_text(new_text)
                self.set_edit_pos(new_pos)
                return

            self.set_caption(("3", "Evaluating... "))

            return super().keypress(size, key)

    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.edit = self.CustomCommandLine(self.parser)

        self.overlay = urwid.Overlay(
            cRoundedLineBox(self.edit, title="Enter Simulation UID", focus_map="8"),
            Caller.main_loop.widget,
            align="center",
            width=100,
            valign="middle",
            height=3,
        )

        urwid.connect_signal(self.edit, "execute_command", self.execute_command)

        super().__init__(self.edit, self.overlay)

    def execute_command(self, command):
        """
        Open the hdf file with the given UID
        """
        show_error = lambda: self.edit.set_caption(("2", "Invalid UID - "))
        if command.split(":")[0] not in self.parser.db_ids:
            show_error()
            return

        try:
            sim_file = Simulation.fromUID(command).h5file
        except (ValueError, FileNotFoundError):
            show_error()
            return

        widget = HDFView(sim_file)
        Caller.enter_widget(widget)
