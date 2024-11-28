from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import threading
from typing import Callable, Generator

import hfive
import pyperclip
import urwid
from bamboost.tui.common import (
    Caller,
    Config,
    redirect_allout,
)
from bamboost.tui.keybinds import Keybindings, apply_user_keybinds
from bamboost.tui.parser.parser import ArgumentParser
from bamboost.tui.widgets import custom_widgets as cw
from bamboost.tui.widgets.autocomplete_edit import AutocompleteContainer
from bamboost.tui.widgets.command_line import CommandLine
from bamboost.tui.widgets.table import RowEntry, Table

import bamboost.core as bb

try:
    sys.modules.pop("gtk")  # florez: this fixes a bug with pyperclip (unknown reason)
except KeyError:
    pass

DIVIDE_CHARS = 2
FOOTER = "q: Exit widget | Q: Quit app | ?: Keybindings | : Command"

database_keybinds = Config.keybinds.get("database", {})


class DatabaseCommands:
    def __init__(self, parent: Database) -> None:
        self.parent = parent

        self.parser = ArgumentParser()  # Enhanced argparse parser

        functions = self.parser.add_subparsers(dest="function")

        # sort
        sort = functions.add_parser("sort", help="Sort by a column")
        sort.add_argument("column", choices=list(self.parent.df.columns))
        sort.add_argument(
            "-r", "--reverse", action="store_true", help="Reverse the sort order"
        )
        sort.add_argument("--close", "-c", help="Sort by the closest value in a column")

        # filter
        filter = functions.add_parser("filter", help="Filter table by a column.")
        filter.add_argument(
            "column", choices=list(self.parent.df.columns), help="Column to filter"
        )
        filter.add_argument("-s", "--as_string", help="Force to filter by string comparison", action="store_true")  # fmt: skip
        filter.add_argument("-eq", "--eq", help="Filter by equality")
        filter.add_argument("-ne", "--ne", help="Filter by inequality")
        filter.add_argument("-gt", "--gt", help="Filter by greater than")
        filter.add_argument("-lt", "--lt", help="Filter by less than")
        filter.add_argument("-ge", "--ge", help="Filter by greater than or equal to")
        filter.add_argument("-le", "--le", help="Filter by less than or equal to")
        filter.add_argument("-c", "--contains", help="Filter by substring")

        # goto
        goto = functions.add_parser("goto", help="Go to a column")
        goto.add_argument(
            "column", choices=list(self.parent.df.columns), help="Column to jump to."
        )

        # reset
        functions.add_parser("reset", help="Reset the table to the original dataframe")

        # pin column
        pin = functions.add_parser("pin", help="Pin a column to the left")
        pin.add_argument(
            "column", choices=list(self.parent.df.columns), help="Column to pin"
        )

        # id to stdout
        _ = functions.add_parser(
            "id_to_stdout",
            help="Print the full ID of the selected simulation to stdout",
        )

    @property
    def _functions(self) -> dict[str, Callable]:
        return {
            "sort": self.parent.table.sort,
            "filter": self.parent.table.filter,
            "goto": self.parent.table.go_to_column,
            "reset": self.parent.table.reset,
            "pin": self.parent.table.pin_column,
            "id_to_stdout": self.parent.id_to_stdout,
        }

    def eval(self, command: str) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        sys.stdout = stdout
        sys.stderr = stderr
        try:
            args = self.parser.parse_string(command)
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        except SystemExit as e:  # argparse raises SystemExit on help display
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            if e.args and e.args[0] == 0:  # help displayed
                _show_help(stdout.getvalue())
                return
            if e.args and e.args[0] == 2:  # error displayed
                urwid.emit_signal(self.parent.command_line, "release_focus")
                self.parent.container.footer = urwid.Text(
                    ("failed", stderr.getvalue().splitlines()[-1])
                )
                return

        _func = args.__dict__.pop("function")
        cb = self._functions[_func]
        try:
            cb(**args.__dict__)
        except (KeyError, ValueError, TypeError) as e:
            urwid.emit_signal(self.parent.command_line, "release_focus")
            self.parent.container.footer = urwid.Text(("failed", f"Error: {e}"))
            return

        urwid.emit_signal(self.parent.command_line, "release_focus")
        self.parent.container.footer = urwid.Text(
            ("success", f"Command executed: {command}")
        )


class Database(AutocompleteContainer):
    """Database frame

    TODO: Add description here
    """

    placehold_footer = urwid.Text(("footer", FOOTER))

    def __init__(self, id: str, db: bb.Manager, *, _remote_name: str = None) -> None:
        self.id = id
        self.db = db
        self._remote_name = _remote_name
        """Optional: If this is a remote database, the name of the remote database."""
        self.df = self.db.df
        self.header = urwid.Pile(
            [
                urwid.Text(("6-bold", f"  UID:   {self.id}")),
                urwid.Text(("5-bold", f"  Path:  {self.db.path}")),
                urwid.Divider(),
            ]
        )
        self.table = Table(self.df)

        self.commands = DatabaseCommands(self)
        self.command_line = CommandLine(self.commands.parser)
        urwid.connect_signal(self.command_line, "execute_command", self.commands.eval)
        urwid.connect_signal(
            self.command_line, "release_focus", self.unfocus_command_line
        )

        self.container = urwid.Frame(
            self.table,
            header=self.header,
            footer=self.placehold_footer,
            focus_part="body",
        )

        # fmt: off
        self.keybinds = (
            Keybindings(self)
            .new("command", ":", Database.enter_command_line, "Enter command")
            .new("search", ",", Database.search, "Search")
            .new("jump", "/", lambda self: Database.enter_command_line(self, caption="jump to column: ", prefix="goto "), "Jump to column",)
            .new("help", "?", Database.show_help, "Show keybindings")
            .new("copy", "y", Database.copy_id, "Copy UID")
            .new("open-paraview", "ctrl o", Database.open_paraview, "Open in paraview")
            .new("open-xdmf-file", ["o", "x"], Database.open_xdmf, "Open XDMF")
            .new("open-output-file", ["o", "o"], Database.open_output_file, "Open output file")
            .new("open-submission-file", ["o", "e"], Database.open_submission_file, "Open submission file")
            .new("open-dir", ["o", "d"], Database.open_sim_directory, "Open directory")
            .new("submit", "ctrl p", Database.submit, "Submit job")
            .new("delete", "d", Database.delete, "Delete simulation")
            .new("note", ["o", "n"], Database.edit_note, "Edit note")
            .new("links", "L", Database.show_links, "Show linked sims")
            .new("stdout", "ctrl t", Database.id_to_stdout, "Print full ID to stdout")
            .new("reload", "R", Database.refresh, "Reload dataframe")
            .new("page-down", "ctrl d", lambda *_: Caller.main_loop.process_input(["page down"]), "Page down")
            .new("page-up", "ctrl u", lambda *_: Caller.main_loop.process_input(["page up"]), "Page up")
            .new("next-search", "n", lambda *_: next(self._search_query), "Next search")
        )

        # fmt: on

        # set the user keybindings
        apply_user_keybinds(self.keybinds, "database")
        self.keybinds.resolve_mapping()

        # include keybinds from table widget
        self.keybinds.merge(self.table.keybinds)

        super().__init__(self.command_line, self.container)

        # catch the release_focus signal from the command line
        urwid.connect_signal(self.table, "enter_file", self.enter_file)
        urwid.connect_signal(self.table, "set_footer", self.set_footer)

    def display_loading_screen(self) -> None: ...

    def set_footer(self, widget):
        self.container.footer = widget

    def _reload_df(self) -> None:
        self.df = self.db.get_view()
        self.table.reset_data(self.df)

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        if key in self.keybinds.keys() and self.container.focus_part == "body":
            self.keybinds.call(size, key)
            return
        self.keybinds.reset_submap()
        return self.container.keypress(size, key)

    def copy_id(self) -> None:
        sim_id = self.table._get_entry_in_focus()["id"]

        if self._remote_name:
            full_id = f"ssh://{self._remote_name}/{self.id}:{sim_id}"
        else:
            full_id = f"{self.id}:{sim_id}"

        pyperclip.copy(full_id)
        self.container.footer = urwid.Text(
            ("footer", f"Copied <{full_id}> to clipboard")
        )

    def open_paraview(self) -> None:
        id = self.table._get_entry_in_focus()["id"]
        self.db[id].open_in_paraview()

    def open_xdmf(self) -> None:
        # Lauch EDITOR to edit the xdmf file
        id = self.table._get_entry_in_focus()["id"]
        xdmf_file = self.db[id].xdmffile
        os.system(f"${{EDITOR:-vi}} {xdmf_file}")
        Caller.main_loop.screen.clear()

    def open_output_file(self) -> None:
        # Lauch EDITOR to edit the output file
        id = self.table._get_entry_in_focus()["id"]
        output_file = self.db[id].files(f"{id}.out")
        if os.path.exists(output_file):
            os.system(f"${{EDITOR:-vi}} {output_file}")
            Caller.main_loop.screen.clear()
        else:
            self.container.footer = urwid.Text(
                ("footer", f"No output file found for {id}")
            )

    def open_submission_file(self) -> None:
        # Lauch EDITOR to edit the submission file
        id = self.table._get_entry_in_focus()["id"]
        submission_file_local = self.db[id].files(f"{id}.sh")
        submission_file_slurm = self.db[id].files(f"sbatch_{id}.sh")
        submission_file = (
            submission_file_local
            if os.path.exists(submission_file_local)
            else submission_file_slurm
        )
        subprocess.run([os.getenv("EDITOR", "vi"), submission_file])
        Caller.main_loop.screen.clear()

    def open_sim_directory(self) -> None:
        """Open the simulation directory in the editor."""
        id = self.table._get_entry_in_focus()["id"]
        sim_dir = self.db[id].path
        subprocess.run([os.getenv("EDITOR", "vi"), sim_dir])
        Caller.main_loop.screen.clear()

    def submit(self) -> None:
        """Submit the job for execution."""
        id = self.table._get_entry_in_focus()["id"]

        def _submit() -> None:
            def target(id) -> None:
                # Redirect stdout to a file if submitted locally
                new_stdout = open(f"bamboost-job-{id}.out", "w")
                with redirect_allout(new_stdout):
                    try:
                        self.db[id].submit(stdout=new_stdout, stderr=new_stdout)
                    except FileNotFoundError as e:
                        self.container.footer = urwid.Text(("footer", f"Error: {e}"))
                        Caller.main_loop.draw_screen()
                        return

                self.container.footer = urwid.Text(("footer", f"Job {id} submitted"))
                self.table._df = self.db.get_view()
                self.table.loc(id).recreate()
                Caller.main_loop.draw_screen()

            threading.Thread(target=target, args=(id,)).start()
            Caller.exit_widget()

        ui = cw.cConfirmDialog(
            Caller.main_loop,
            f"Submitting job <{id}>. This will overwrite any existing data. Are you sure?",
            callback=_submit,
        )
        Caller.enter_widget(ui)

    def id_to_stdout(self, id: str = None) -> None:
        """Print the full ID of the selected simulation to stdout."""
        # get the ID of the selected simulation
        id = self.table._get_entry_in_focus()["id"]

        # print the full ID to stdout
        print(self.db[id].get_full_uid(), file=sys.stdout)
        sys.stdout.flush()

        # set the footer for the user
        self.set_footer(
            urwid.Text(
                ("footer", f"Full ID of {id} printed to stdout [Exit app to continue]")
            )
        )

        # exit the app
        # raise urwid.ExitMainLoop()

    def delete(self) -> None:
        """Delete the selected simulation."""
        id = self.table._get_entry_in_focus()["id"]

        def _delete() -> None:
            self.db.remove(id)
            self.table.reset_data(self.db.get_view())

        ui = cw.cConfirmDialog(
            Caller.main_loop,
            f"Are you sure you want to delete {id}?",
            callback=_delete,
        )
        Caller.enter_widget(ui)

    def refresh(self) -> None:
        self._reload_df()

    def edit_note(self) -> None:
        id = self.table._get_entry_in_focus()["id"]
        note = self.db[id].metadata.get("notes", "")

        def commit_note(note: str) -> None:
            self.db[id].change_note(note)
            Caller.exit_widget()
            self.table.loc(id).edit_cell_text("notes", note)

        edit = cw.cEdit(
            edit_text=note, multiline=True, edit_pos=len(note), callback=commit_note
        )
        overlay = urwid.Overlay(
            cw.cRoundedLineBox(
                urwid.Frame(
                    urwid.Filler(urwid.Padding(edit, left=1, right=1), valign="top"),
                    footer=urwid.Pile(
                        [
                            urwid.Divider("\u2500"),
                            urwid.Padding(
                                urwid.Text(("2", "Enter: Save | Esc: Cancel")),
                                left=1,
                                right=1,
                            ),
                        ]
                    ),
                ),
                focus_map="8",
                title="Edit note",
                title_align="left",
            ),
            Caller.widget_stack[-1],
            align="center",
            width=("relative", 80),
            valign="middle",
            height=20,
        )
        Caller.enter_widget(overlay)

    def show_links(self) -> None:
        """Show the linked simulations."""
        id = self.table._get_entry_in_focus()["id"]
        links = self.db[id].links.all_links()

        def cb(action_item: cw.cActionItem):
            id = action_item.content[1]
            widget = hfive.HFive(bb.Simulation.fromUID(id).h5file)
            Caller.enter_widget(widget)

        ui = cw.cPopup(
            cw.cListBoxSelectionCharacter(
                [cw.cActionItem([key, val], callback=cb) for key, val in links.items()],
                attr_map={0: "5", 1: "8"},
                focus_map={0: "5-bold", 1: "8-bold"},
            ),
            footer="Enter: Open | Esc: Close",
            title="Linked simulations",
            title_align="left",
            focus_map="8",
            height=len(links),
        )
        Caller.enter_widget(ui)

    def enter_command_line(
        self,
        *args,
        caption: str | None = None,
        prefix: str | None = None,
        container: urwid.Widget | None = None,
        **kwargs,
    ) -> None:
        self.command_line.set_caption(caption or ":")
        self.commands.parser.set_prefix(prefix or "")

        self.container.footer = self.command_line
        self.command_line.edit_text = ""
        self.command_line.history_index = 0
        self.container.set_focus("footer")

    @property
    def _search_query(self) -> None:
        return self.__search_gen

    @_search_query.setter
    def _search_query(self, value):
        self.__search_gen = value

    def _search(self, text: str) -> Generator[None, None, None]:
        rows: list[RowEntry] = self.table.entries.body
        for row in rows:
            for col_name, cell in row._cells.items():
                cell_w = cell[2].base_widget
                if cell_w.text.lower().find(text.lower()) != -1:
                    # split the text into two parts (before and after the match) using regex module
                    text_orig = cell_w.text
                    before, match, after = re.split(
                        f"({text})", cell_w.text, flags=re.IGNORECASE
                    )
                    cell_w.set_text([before, ("match", match), after])
                    self.table.go_to_column(col_name)
                    yield
                    # reset the text to the original
                    cell_w.set_text(text_orig)

    def search(self) -> None:
        """Search for string in the table and highlight it."""

        def search(text):
            self._search_query = self._search(text)
            self.unfocus_command_line()
            next(self._search_query)

        edit = cw.cEdit(caption="search: ", callback=search)
        self.container.footer = edit
        self.container.set_focus("footer")
        return

    def unfocus_command_line(self) -> None:
        self.container.footer = self.placehold_footer
        self.container.set_focus("body")

    def enter_file(self, id: str) -> None:
        Caller.enter_widget(hfive.HFive(self.db[id].h5file))

    def show_help(self) -> None:
        # if not hasattr(self, "_help_widget"):
        #     self._help_widget = cKeybindsOverlay(Caller.main_loop, self.keybinds)
        # self._help_widget.toggle()
        self.keybinds.toggle_help()


def _show_help(text) -> None:
    class HelpFrame(urwid.Frame):
        def keypress(self, size, key: str) -> str | None:
            if key in {"q", "esc"}:
                Caller.exit_widget()
                return
            return super().keypress(size, key)

    Caller.enter_widget(
        HelpFrame(
            urwid.Overlay(
                cw.cRoundedLineBox(
                    urwid.Text(text), focus_map="8", title="Help", title_align="left"
                ),
                Caller.main_loop.widget,
                "left",
                ("relative", 100),
                "bottom",
                ("pack"),
                bottom=1,
            )
        )
    )
