#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

os.environ["BAMBOOST_MPI"] = "0"

import urwid
from bamboost import __version__
from bamboost.tui.common import Caller, palette

import bamboost


def main():
    if not sys.stdin.isatty():
        path = input()
        sys.stdin = open("/dev/tty", "r")
    else:
        parser = argparse.ArgumentParser(description="bamboost.tui")
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"bamboost {bamboost.__version__}; bamboost.tui {__version__}",
        )
        parser.add_argument(
            "-r",
            "--remote",
            action="store_true",
            help="Directly open default remote database.",
        )
        parser.add_argument(
            "-l",
            "--locking",
            action="store_true",
            help="Enable custom locking of h5 files.",
        )
        parser.add_argument(
            "path", nargs="?", help="Path to directly open a bamboost database."
        )
        args = parser.parse_args()
        path = args.path

        if args.locking:
            from bamboost.core.extensions.use_locking import use_locking

            use_locking("exclusive")

    if sys.stdout.isatty():
        Caller.OUTPIPE = False
    else:
        Caller.OUTPIPE = True

    if args.remote:
        from bamboost.tui.widgets.custom_widgets import cPopup
        from bamboost.tui.widgets.index import IndexRemote, Remote

        first_remote = Remote.list()[0]
        remote_index = IndexRemote(remote=Remote(first_remote, skip_update=True))
        ui = cPopup(
            remote_index,
            title=f"Remote databases [{first_remote}] - ctrl + r to fetch index",
        )
        enter_main_loop(ui)
        return

    if path is not None:
        from bamboost.tui.pages.db import Database
        from bamboost.tui.widgets.custom_widgets import (
            cActionItem,
            cListBoxSelectionCharacter,
            cPopup,
        )

        from bamboost.core import IndexAPI, Manager

        # Filter the databases which have the given path as a parent
        # If only 1 database is found, open it directly
        # If multiple databases are found, show a list of them
        path = os.path.abspath(path)
        index = IndexAPI().fetch("SELECT id, path FROM dbindex")
        candidates = {
            id: str(Path(val).relative_to(Path(path)))
            for id, val in index
            if Path(path) in [*Path(val).parents, Path(val)]
        }

        if len(candidates) == 1:
            id = next(iter(candidates))
            ui = Database(id, Manager(uid=id))
            enter_main_loop(ui)
            return

        if len(candidates) > 1:

            def cb(action_item: cActionItem):
                id = action_item.content[0]
                ui = Database(id, Manager(uid=id))
                Caller.widget_stack.pop()
                Caller.enter_widget(ui)

            items = [
                cActionItem(candidate, callback=cb) for candidate in candidates.items()
            ]
            ui = cPopup(
                cListBoxSelectionCharacter(
                    items,
                    attr_map={0: "5", 1: "8"},
                    focus_map={0: "5-bold", 1: "8-bold"},
                ),
                height=len(candidates),
                title="Select a database",
            )
            enter_main_loop(ui)
            return

    from bamboost.tui.pages.welcome import WelcomeUI

    enter_main_loop(WelcomeUI().frame)


def enter_main_loop(widget, screen=None):
    screen = urwid.raw_display.Screen(input=sys.stdin, output=sys.stderr)
    Caller.main_loop = urwid.MainLoop(
        widget, palette, unhandled_input=exit, screen=screen
    )
    Caller.main_loop.handle_mouse = True
    Caller.main_loop.screen.set_terminal_properties(colors=256)
    Caller.widget_stack.append(widget)
    Caller.main_loop.run()


def exit(key):
    if key == "q":
        Caller.exit_widget()
    elif key == "Q":
        raise urwid.ExitMainLoop()


if __name__ == "__main__":
    main()
