"""This module contains common classes and functions used by the TUI
application.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from itertools import chain

import urwid
from bamboost.tui import tomllib

# Constants
FIGLET = """
██████╗  █████╗ ███╗   ███╗██████╗  ██████╗  ██████╗ ███████╗████████╗
██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔═══██╗██╔═══██╗██╔════╝╚══██╔══╝
██████╔╝███████║██╔████╔██║██████╔╝██║   ██║██║   ██║███████╗   ██║   
██╔══██╗██╔══██║██║╚██╔╝██║██╔══██╗██║   ██║██║   ██║╚════██║   ██║   
██████╔╝██║  ██║██║ ╚═╝ ██║██████╔╝╚██████╔╝╚██████╔╝███████║   ██║   
╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   
"""
VERTICAL_KEYS = {
    "up",
    "down",
    "ctrl n",
    "ctrl p",
    "page up",
    "page down",
    "home",
    "end",
}


class Caller:
    """Caller holds the main urwid loop and a widget stack. It manages the flow
    of pages and widgets, e.g. which is on top and which should be shown when a
    widget is exited.

    Attributes:
        widget_stack: A list of widgets that are currently in the stack.
        main_loop: The main loop of the urwid application.
    """

    widget_stack = []
    main_loop: urwid.MainLoop = None

    @classmethod
    def enter_widget(cls, widget: urwid.Widget):
        """Enters a new widget into the widget stack and sets it as the main
        widget in the main loop.

        Args:
            widget (urwid.Widget): The widget to enter into the widget stack.
        """
        cls.widget_stack.append(widget)
        cls.main_loop.widget = widget

    @classmethod
    def exit_widget(cls):
        """Exits the current widget by removing it from the widget stack and
        setting the previous widget as the main widget in the main loop.

        Raises:
            urwid.ExitMainLoop: If there are no more widgets in the stack,
                raises an ExitMainLoop exception to exit the main loop.
        """
        # If no widgets are in the stack, exit the main loop
        if len(cls.widget_stack) <= 1:
            raise urwid.ExitMainLoop()

        # Set the widget below as the main widget
        cls.widget_stack.pop()
        cls.main_loop.widget = cls.widget_stack[-1]


# Add Caller to urwid module for shared access with subapp HFive
# This is a hack to share the Caller object with the subapp HFive
# Let me know if you have a better idea
urwid.Caller = Caller


# ------------------------------------------
# CONFIG
# ------------------------------------------
CONFIG_DIR = os.path.expanduser("~/.config/bamboost")
CONFIG_FILE = os.path.join(CONFIG_DIR, "tui.toml")
CUSTOM_FUNCTIONS = os.path.join(CONFIG_DIR, "custom_functions.py")


class Config:
    """
    Namespace for configuration settings from the user config file
    "~/.config/bamboost/tui.toml".

    Attributes:
        keybinds: Parsed dictionary of the [keybinds] table from the config
            file.
        theme: The theme to use for the application. (Not used yet)
        custom_functions: A namespace for custom functions defined in the
            custom functions file.
    """

    keybinds: dict[str, str] = {}
    theme: str = "default"
    custom_functions: dict = {}
    table: dict = {}

    @classmethod
    def load_custom_functions(cls):
        """Read the custom functions file and execute it to define the custom
        functions. The global namespace of the custom functions file is set to
        the `custom_functions` dictionary of this class.
        """
        for file in chain(cls.keybinds.get("custom_files", []), [CUSTOM_FUNCTIONS]):
            if not os.path.exists(file):
                continue
            with open(file, "r") as f:
                custom_functions = f.read()
            exec(custom_functions, cls.custom_functions)


# Read the configuration file at import time
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)
    Config.keybinds.update(config.get("keybinds", {}))
    Config.theme = config.get("theme", "default")
    # Option to not clip certain columns if they are too long
    Config.table = config.get("table", {})


# ------------------------------------------
# COLOR PALETTE
# ------------------------------------------
# 1-16: standard colors
color_map = {
    1: "black",
    2: "dark red",
    3: "dark green",
    4: "brown",  # or 'yellow' for some terminals that treat this as yellow/brown/orange
    5: "dark blue",
    6: "dark magenta",
    7: "dark cyan",
    8: "light gray",
    9: "dark gray",
    10: "light red",
    11: "light green",
    12: "yellow",  # or 'brown' if the terminal uses yellow for color 4
    13: "light blue",
    14: "light magenta",
    15: "light cyan",
    16: "white",
}

palette: list = []
"""list: A list of tuples defining the color palette for the application.

There are color variants for the standard 16 colors (denoted as C below).
C: just the color
C-bold: the color with bold text
C-reverse: the color as background and black text
C-reverse-bold: the color as background with bold black text
C-on-white: the color with white background
C-bold-on-white: the color with bold text and white background
"""
# Add default colors to palette 1-16
palette.extend([(str(num), "", "", "", color_map[num], "") for num in color_map])
# bold
palette.extend(
    [(f"{num}-bold", "", "", "", f"{color_map[num]},bold", "") for num in color_map]
)
# reverse
palette.extend([(f"{num}-reverse", "black", color_map[num]) for num in color_map])
# bold-reverse
palette.extend(
    [(f"{num}-reverse-bold", "black,bold", f"{color_map[num]}") for num in color_map]
)
# on-white
palette.extend(
    [
        (
            f"{num}-on-white",
            f"{color_map[num] if num<=8 else color_map[num-8]}",
            "white",
        )
        for num in color_map
    ]
)
# bold-on-white
palette.extend(
    [
        (
            f"{num}-bold-on-white",
            f"{color_map[(num-1) % 8 + 1]},bold",
            "white",
            "",
            f"{color_map[(num-1) % 8 + 1]},bold",
            "h15",
        )
        for num in color_map
    ]
)
# Add some custom definitions
palette.extend(
    [
        ("selected", "standout", ""),
        ("bold", "bold", ""),
        ("boldselected", "standout,bold", ""),
        ("footer", color_map[2], ""),
        ("green_box", color_map[6], ""),
        ("default text", color_map[2], ""),
        ("failed", "dark red", ""),
        ("success", "dark green", ""),
        ("reverse", "black", "white"),
        ("match", "black", "yellow"),
    ]
)


@contextmanager
def redirect_stdout(new_stdout="devnull"):
    if new_stdout == "devnull":
        new_stdout = open(os.devnull, "w")
    original_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        new_stdout.close()
        sys.stdout = original_stdout


@contextmanager
def redirect_allout(new_stdout="devnull"):
    if new_stdout == "devnull":
        new_stdout = open(os.devnull, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = new_stdout
    sys.stderr = new_stdout
    try:
        yield None
    finally:
        new_stdout.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class Spinner(urwid.AttrMap):
    def __init__(self, attr_map: str = None, focus_map: str = None, **kwargs):
        self.spinner = ["|", "/", "-", "\\"]
        self.frame = 0
        self.on = False
        self.spinner_speed = kwargs.get("speed", 0.1)
        self.caption = kwargs.get("caption", "")
        super().__init__(
            urwid.Text("", align=kwargs.get("align", "left")), attr_map, focus_map
        )

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def update_spinner(self, loop, user_data=None):
        if self.on:
            self.frame = (self.frame + 1) % len(self.spinner)
            self.base_widget.set_text("".join((self.caption, self.spinner[self.frame])))
            loop.set_alarm_in(self.spinner_speed, self.update_spinner)

    def start(self) -> Spinner:
        self.on = True
        self.update_spinner(Caller.main_loop)
        return self

    def stop(self) -> Spinner:
        self.base_widget.set_text("")
        self.on = False
        return self
