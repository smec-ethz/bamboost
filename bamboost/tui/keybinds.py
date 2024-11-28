"""Keybind module of the bamboost tui.

This module contains the Keybind and the Keybindings class. `Keybind`
represents a single keybinding, while `Keybindings` is a collection of
keybindings for a specific page or widget.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Generator

import urwid
from bamboost.tui.common import Caller, Config
from bamboost.tui.widgets.rounded_line_box import cRoundedLineBox


def apply_user_keybinds(keybinds: Keybindings, table: str) -> None:
    """Apply user keybindings to the keybindings table.

    Args:
        keybinds (Keybindings): The keybindings table.
        table (str): The corresponding table in the toml config. E.g.
            "database", ...
    """
    Config.load_custom_functions()
    for name, key in Config.keybinds.get(table, {}).items():
        if isinstance(key, dict):
            new_keybind = Keybind(
                name,
                key["key"],
                Config.custom_functions[key["func"]],
                "",
                group="custom",
            )
            keybinds.add(new_keybind)
            continue

        # if just a key is given, use the default function but replace the key
        keybinds[name].key = key


@dataclass
class Keybind:
    """A single keybinding.

    Args:
        name: The name of the keybinding.
        key: The key or keys that trigger the keybinding.
        func: The function that is called when the keybinding is triggered.
        description: A short description of the keybinding.
        aliases: Optional. A list of aliases for the keybinding.
        group: Optional. The group to which the keybinding belongs. This is used to group
            the help widget.
        obj: Optional. The object (instance) to which the keybinding belongs to. This is
            passed as the first argument to the function.
    """

    name: str
    key: str | list[str]
    func: callable
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    group: str = None
    obj: object = None

    def key_as_string(self) -> str:
        """Return the key as a string. If the key is a list, it is joined by '  '."""
        if isinstance(self.key, list):
            return "  ".join(self.key)
        return self.key

    def call(self, size: tuple[int, int], key: str) -> None:
        """Call the function associated with the keybinding."""
        if self.obj is None:
            return self.func(size, key)

        # check if the function takes the size and key as arguments
        if self.func.__code__.co_argcount == 1:
            return self.func(self.obj)
        return self.func(self.obj, size, key)


class Keybindings(MutableMapping[str, Keybind]):
    def __init__(self, obj, *, group: str = None) -> None:
        """
        Args:
            obj: The object to which the keybindings are applied.
            group: The group to which the keybindings belong.
        """
        self._obj = obj
        self._map: dict[str, Keybind | dict] = {}
        self._keychain = []
        self._submap: dict[str, Keybind] = None
        self._store: dict[str, Keybind] = {}
        self.group = group

    def add(self, new: Keybind) -> None:
        self[new.name] = new

    def resolve_mapping(self) -> None:
        """Resolve the mapping of the keybindings.

        This method is called after all keybindings have been added to the
        keybindings table. It resolves the mapping of the keybindings and
        creates a dictionary of the available keys (and aliases) and their
        functions.
        """
        # handle sub-maps
        for key in self._store.values():
            if isinstance(key.key, list):
                submap = self._map
                for i in key.key[:-1]:
                    if i not in submap:
                        submap[i] = {}
                    submap = submap[i]
                submap[key.key[-1]] = key
            else:
                try:
                    self._map.update({key.key: key})
                except TypeError:
                    for key in self._store.values():
                        print(key.name, key.key)
                    raise SystemExit

            self._map.update({alias: key for alias in key.aliases})
        self._submap = self._map

    def new(
        self,
        name: str,
        key: str | list[str],
        func: callable,
        description: str = "",
        aliases: list[str] = None,
    ) -> Keybindings:
        """Add a new keybinding to the keybindings table.

        Adds a new keybinding with default values for the group and object.

        Args:
            name: The name of the keybinding.
            key: The key or keys that trigger the keybinding.
            func: The function that is called when the keybinding is triggered.
            description: Optional. A short description of the keybinding.
            aliases: Optional. A list of aliases for the keybinding.
        """
        if aliases is None:
            aliases = []
        self.add(
            Keybind(
                name, key, func, description, aliases, obj=self._obj, group=self.group
            )
        )
        return self

    def keys(self) -> dict[str, callable]:
        """Return a dictionary of the available keys and their functions."""
        return self._submap.keys()

    def call(self, size: tuple[int, int], key: str) -> None:
        """Call the function associated with the given key or enter submap.

        Args:
            key: The key to pursue.
        """
        if isinstance(self._submap[key], dict):
            self.update_submap(key)
            return

        self.key_indicator_widget.reset()
        self._submap[key].call(size, key)
        self.reset_submap()

    def merge(self, other: Keybindings, group: str = None) -> None:
        """Merge the keybindings of another Keybindings object into this one.

        Args:
            other: The other Keybindings object.
            group: Optional. The group to which the keybindings belongs.
        """
        # merge this keybindings into the other first to overwrite duplicates correctly
        tmp = other._store.copy()
        # change group
        if group is not None:
            for key in tmp.values():
                key.group = group
        tmp.update(self._store)
        self._store.update(tmp)
        self.resolve_mapping()

    @property
    def key_indicator_widget(self) -> KeyIndicatorWidget:
        """Return the key indicator widget.

        If a keychain is pressed, the key indicator prints the current keychain
        to the top right of the screen.
        """
        if not hasattr(self, "_key_indicator_widget"):
            self._key_indicator_widget = KeyIndicatorWidget(
                Caller.main_loop, Caller.widget_stack[-1], ""
            )
        return self._key_indicator_widget

    def update_submap(self, key: str) -> None:
        """Update the submap to the submap of the given key."""
        self._submap = self._submap[key]
        self._keychain.append(key)
        self.key_indicator_widget.set_text(" > ".join(self._keychain))
        if not self.key_indicator_widget.is_visible:
            self.key_indicator_widget.toggle()

    def reset_submap(self) -> None:
        """Reset the submap to the main map."""
        self._submap = self._map
        self._keychain = []

    def toggle_help(self) -> None:
        """Toggle the keybinds overlay."""
        if not hasattr(self, "_keybinds_overlay"):
            self._keybinds_overlay = cKeybindsOverlay(
                Caller.main_loop, Caller.widget_stack[-1], self
            )
        self._keybinds_overlay.toggle()

    def __getitem__(self, key: str) -> Keybind:
        "Get a Keybind by name."
        return self._store[key]

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __setitem__(self, key: str, value: Keybind) -> None:
        "Set a Keybind by name."
        self._store[key] = value

    def __iter__(self) -> Generator[Keybind, None, None]:
        "Iterate over the Keybinds stored here."
        return iter(self._store.values())

    def __len__(self) -> int:
        return len(self._store)


class KeyIndicatorWidget(urwid.Frame):
    """Displays the current keychain in an overlay.

    Args:
        main_loop (urwid.MainLoop): The main loop (Caller.main_loop)
        bottom_w (urwid.Widget): The widget that is displayed below the
            overlay. Usually this should be `Caller.widget_stack[-1]`
        text: Initial text. Default is an empty string.
    """

    is_visible: bool = False

    def __init__(
        self, main_loop: urwid.MainLoop, bottom_w: urwid.Widget, text: str = ""
    ):
        self.main_loop = main_loop
        self.bottom_w = bottom_w
        self.text_w = urwid.AttrMap(urwid.Text(text), "8", "8")

        self.overlay = urwid.Overlay(
            self.text_w,
            self.bottom_w,
            align="right",
            width="pack",
            valign="bottom",
            height=1,
        )
        super().__init__(self.overlay)

    def set_text(self, text: str) -> None:
        self.text_w.base_widget.set_text(text)

    def reset(self) -> None:
        self.text_w.base_widget.set_text("")
        if self.is_visible:
            self.toggle()

    def toggle(self) -> None:
        if self.is_visible:
            self.main_loop.widget = self.bottom_w
            self.is_visible = False
        else:
            self.main_loop.widget = self
            self.is_visible = True

    def keypress(
        self, size: tuple[()] | tuple[int] | tuple[int, int], key: str
    ) -> str | None:
        # this widget should pass all keypresses to the widget below
        return self.bottom_w.keypress(size, key)


class cKeybindsOverlay(urwid.Frame):
    """Displays the keybindings in an overlay

    Args:
        main_loop (urwid.MainLoop): The main loop (Caller.main_loop)
        keybinds (Keybinds): The keybindings to be displayed
    """

    _sizing = frozenset(["box"])
    is_visible: bool = False

    def __init__(
        self,
        main_loop: urwid.MainLoop,
        bottom_w: urwid.Widget,
        keybinds: Keybindings | tuple[Keybindings],
        *args,
        **kwargs,
    ):
        self.main_loop = main_loop
        self.bottom_w = bottom_w
        self.keybinds: Keybindings = keybinds

        # group the keybinds by group
        groups = {}
        for key in keybinds:
            if key.group not in groups:
                groups[key.group] = []
            groups[key.group].append(key)

        # calculate the longest name and key for padding
        longest_name = max(len(key.name) for key in keybinds)
        longest_key = max(len(key.key_as_string()) for key in keybinds)

        # construct the list entries for the pile
        list_entries = []
        for group, keys in groups.items():
            if group:
                list_entries.append(urwid.Divider("-"))
                list_entries.append(
                    urwid.Padding(
                        urwid.Text(("8-bold", group.capitalize()), align="center")
                    ),
                )
            list_entries.extend(
                [
                    urwid.Columns(
                        [
                            (longest_name, urwid.Text(("4", key.name), align="left")),
                            (
                                longest_key,
                                urwid.Text(
                                    ("3-bold", key.key_as_string()), align="right"
                                ),
                            ),
                            (
                                "pack",
                                urwid.Padding(
                                    urwid.Text(("", key.description)),
                                    width=min(
                                        Caller.main_loop.screen.get_cols_rows()[0] // 3,
                                        len(key.description),
                                    ),
                                ),
                            ),
                            (
                                "weight",
                                1,
                                urwid.Text(
                                    (
                                        "3",
                                        "[" + ", ".join(key.aliases) + "]"
                                        if key.aliases
                                        else "",
                                    ),
                                    align="right",
                                ),
                            ),
                        ],
                        dividechars=2,
                    )
                    for key in keys
                ]
            )

        # calculate the max width and height of the columns for the sizing of the overlay
        max_width = max(
            w.pack(())[0] for w in list_entries if urwid.FIXED in w.sizing()
        )
        max_height = sum(
            w.pack((max_width,))[1] for w in list_entries if urwid.FIXED in w.sizing()
        ) + sum(1 for w in list_entries if urwid.FIXED not in w.sizing())

        self.box = cRoundedLineBox(
            urwid.Padding(
                urwid.ListBox(urwid.SimpleFocusListWalker(list_entries)),
                left=1,
                right=1,
            ),
            focus_map="8",
            title="Keybinds",
            title_align="left",
        )
        self.overlay = urwid.Overlay(
            self.box,
            self.bottom_w,
            align="left",
            valign="bottom",
            height=max_height + 2,  # +2 for border
            width=max_width + 4,  # +2 for padding, +2 for border
            bottom=1,
        )
        super().__init__(self.overlay, *args, **kwargs)

    def toggle(self) -> None:
        if self.is_visible and self.main_loop.widget == self:
            self.main_loop.widget = self.bottom_w
            self.is_visible = False
        else:
            self.main_loop.widget = self
            self.is_visible = True

    def keypress(
        self, size: tuple[()] | tuple[int] | tuple[int, int], key: str
    ) -> str | None:
        if key in ("esc", "q"):
            # Set the widget of loop back to the original widget
            return self.toggle()

        # all other keypresses are passed to the widget below
        return self.bottom_w.keypress(size, key)
