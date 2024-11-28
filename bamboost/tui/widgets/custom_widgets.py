from __future__ import annotations

from typing import Callable, Iterable, List, Tuple, Union

import urwid
from bamboost.tui.common import Caller
from bamboost.tui.keybinds import Keybindings
from bamboost.tui.widgets.rounded_line_box import cRoundedLineBox

SELECT_CHARACTER = "❯"


class SelectionIcon(urwid.Text):
    def __init__(self):
        self.icon = SELECT_CHARACTER
        super().__init__(self.icon, align="center")

    def render(self, size, focus=False):
        if focus:
            self.set_text(("4", self.icon))
        else:
            self.set_text(" ")
        return super().render(size, focus)


class cListBox(urwid.ListBox):
    _sizing = frozenset(["box", "flow"])

    def __init__(self, *args, wrap: bool = False, keymap_jk: bool = False, **kwargs):
        self.wrap = wrap
        self.keymap_jk = keymap_jk
        super().__init__(*args, **kwargs)

        # fmt: off
        self.keybinds = (
            Keybindings(self)
            .new("down", "j", cListBox._navigate_down, "Move focus down", ["down", "ctrl n"])
            .new("up", "k", cListBox._navigate_up, "Move focus up", ["up", "ctrl p"])
            .new("start", ["g", "g"], lambda self: self.set_focus(0), "Move focus to the start")
            .new("end", "G", lambda self: self.set_focus(len(self.body) - 1), "Move focus to the end")
        )
        # fmt: on
        self.keybinds.resolve_mapping()

    def _navigate_down(self, *args) -> None:
        try:
            self.set_focus(self.focus_position + 1, "above")
        except IndexError:
            pass

    def _navigate_up(self, *args) -> None:
        try:
            if self.focus_position > 0:
                self.set_focus(self.focus_position - 1, "below")
        except IndexError:
            pass

    def rows(self, size, focus):
        return len(self.body)

    @property
    def focused_text(self) -> str:
        return self.get_focus()[0].base_widget.text

    def keypress(self, size: tuple, key: str) -> Union[str, None]:
        from bamboost.tui.keybinds import cKeybindsOverlay

        if key == "?":
            if not hasattr(self, "_help_widget"):
                self._help_widget = cKeybindsOverlay(
                    Caller.main_loop, Caller.widget_stack[-1], self.keybinds
                )
            self._help_widget.toggle()
            return

        if key in self.keybinds.keys():
            self.keybinds.call(size, key)
            return

        self.keybinds.reset_submap()
        return super().keypress(size, key)


class cEdit(urwid.Edit):
    signals = ["release_focus", "execute_command", "set_footer"]

    def __init__(self, *args, callback: callable = None, **kwargs):
        self.callback = callback
        super().__init__(*args, **kwargs)

    def keypress(self, size: tuple, key: str) -> Union[str, None]:
        if key == "meta backspace":
            # Logic to delete the word before the cursor
            new_pos = (
                self.edit_text.rfind(" ", 0, self.edit_pos - 1) + 1
                if self.edit_pos > 0
                else 0
            )
            new_text = self.edit_text[:new_pos] + self.edit_text[self.edit_pos :]
            self.set_edit_text(new_text)
            self.set_edit_pos(new_pos)
        elif key == "esc":
            Caller.exit_widget()
        elif key == "enter":
            if self.callback is not None:
                self.callback(self.edit_text)
                return
            return super().keypress(size, key)

        else:
            return super().keypress(size, key)


class cEditHistory(urwid.Edit):
    def __init__(self, *args, **kwargs):
        self.history = [""]
        self.history_index = 0
        super().__init__(*args, **kwargs)

    @property
    def history_index(self) -> int:
        return self._history_index

    @history_index.setter
    def history_index(self, value: int) -> None:
        if value < 0:
            self._history_index = 0
        elif value > len(self.history):
            self._history_index = len(self.history)
        else:
            self._history_index = value

    def keypress(self, size: tuple[int], key: str) -> str | None:
        if key in {"up", "ctrl p"}:
            # Move up the history
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                text = self.history[-self.history_index]
                self.set_edit_text(text)
                self.set_edit_pos(len(text))
        elif key in {"down", "ctrl n"}:
            # Move down the history
            self.history_index -= 1
            text = self.history[-self.history_index]
            self.set_edit_text(text)
            self.set_edit_pos(len(text))
        else:
            return super().keypress(size, key)


class cConfirmDialog(urwid.Frame):
    """A dialog box for confirming an action.

    Args:
        - main_loop (urwid.MainLoop): The main loop (Caller.main_loop)
        - text (str): The text to display in the dialog box.
        - callback (callable): The callback function to be called when the "Yes"
          button is pressed.
    """

    _selectable = True

    def __init__(
        self, main_loop: urwid.MainLoop, text: str, callback: callable, *args, **kwargs
    ):
        self.main_loop = main_loop
        self.bottom_w = main_loop.widget
        self.text = text
        self.callback = callback
        self.yes_button = urwid.Button("Yes", on_press=self.yes)
        self.no_button = urwid.Button("No", on_press=self.no)
        self.buttons = urwid.Columns([self.no_button, self.yes_button], dividechars=2)
        self.box = cRoundedLineBox(
            urwid.Pile([urwid.Text(text), self.buttons]),
            focus_map="2",
            title="Confirmation",
        )
        self.overlay = urwid.Overlay(
            self.box,
            self.bottom_w,
            align="center",
            width="pack",
            valign="middle",
            height="pack",
        )
        super().__init__(self.overlay, *args, **kwargs)

    def yes(self, button: urwid.Button) -> None:
        Caller.exit_widget()
        self.callback()
        return

    def no(self, button: urwid.Button) -> None:
        Caller.exit_widget()
        return

    def keypress(
        self, size: tuple[()] | tuple[int] | tuple[int, int], key: str
    ) -> str | None:
        if key in ("esc", "q"):
            # Set the widget of loop back to the original widget
            Caller.exit_widget()
            return
        if key == "l":
            return super().keypress(size, "right")
        if key == "h":
            return super().keypress(size, "left")

        return super().keypress(size, key)


class cActionItem(urwid.Widget):
    """A custom widget class for action items in an Urwid application.

    Args:
        - content (str | Iterable): The content to be displayed in the widget.
          Can be a list of strings.
        - callback (callable, optional): The callback function to be called
          when the widget is activated. The callback function takes the widget
          as an argument.
    """

    _sizing = frozenset(["box"])
    _selectable = True
    signals = ["set_footer"]

    DIVIDECHARS = 2

    def __init__(self, content: str | Iterable, callback: callable = None) -> None:
        self.content = [content] if isinstance(content, str) else content
        self._callback = callback
        self.column_widths = {}
        self.column_attr_map = {None: "default"}
        self.column_focus_map = {None: "bold"}

        self._invalidate()
        super().__init__()

    def _invalidate(self):
        self.widget = urwid.AttrMap(
            urwid.Columns(
                [("fixed", 1, SelectionIcon())]
                + [self._render_column(i, text) for i, text in enumerate(self.content)],
                dividechars=self.DIVIDECHARS,
            ),
            self.column_attr_map,
            self.column_focus_map,
        )
        return super()._invalidate()

    def rows(self, size, focus):
        return self.widget.rows(size, focus)

    def callback(self) -> Callable:
        return self._callback(self) or (lambda *args: None)

    def keypress(self, size, key):
        if key == "enter":
            self.callback()
            return
        return key

    @property
    def widget(self) -> urwid.Columns:
        return self._widget

    @widget.setter
    def widget(self, value: urwid.Columns) -> None:
        self._widget = value

    def render(self, size, focus=False):
        # Before render, the rows() method is called by urwid!
        # Therefore, we don't need to update the widget here again but only in rows()
        return self.widget.render(size, focus)

    def _render_column(self, i: int, text: str, focus: bool = False):
        sizing = urwid.WEIGHT
        if hasattr(self, "_column_sizing") and self._column_sizing:
            sizing = self._column_sizing[i]

        align = urwid.LEFT
        if hasattr(self, "_column_align") and self._column_align:
            align = self._column_align[i]

        if self.column_widths and (i == 0 or sizing == urwid.FIXED):
            return (
                "fixed",
                self.column_widths[i],
                urwid.AttrMap(urwid.Text(text, align=align), i),
            )

        if sizing == urwid.PACK:
            return (sizing, urwid.AttrMap(urwid.Text(text, align=align), i))

        return (
            "weight",
            1,
            urwid.AttrMap(urwid.Text(text, align=align, wrap="clip"), i),
        )


class cListBoxSelectionCharacter(cListBox):
    """A custom list box widget for selecting from a list of items with a focus
    character.

    Args:
        - items (List[cActionItem]): A list of cActionItem objects representing
          the items in the list box.
        - attr_map (List[Tuple[str]]): A list of tuples specifying the
          attribute mappings for columns. Defaults to None.
        - always_in_focus (bool): Whether the list box should always be in
          focus. Defaults to False.
        - headers (List[str]): A list of strings specifying the headers for the
          columns to calculate the column widths. Defaults to None.

    Kwargs:
        - column_sizing (List[str]): A list of strings specifying the sizing
          of the columns. Defaults to None.
        - column_align (List[str]): A list of strings specifying the alignment
          of the columns. Defaults to None.

    Attributes:
        - _column_widths (Dict[int, int]): A dictionary mapping column index to
          the width of the longest content in that column. items
          (List[cActionItem]): A list of cActionItem objects representing the
          items in the list box.
        - _listwalker (urwid.SimpleFocusListWalker): A SimpleFocusListWalker
          object for navigating the list items.
    """

    def __init__(
        self,
        items: List[cActionItem],
        attr_map: dict = None,
        focus_map: dict = None,
        always_in_focus: bool = False,
        headers: List[str] = None,
        *args,
        **kwargs,
    ):
        # get the length of the longest key per column
        self._column_widths = {}
        for item in items:
            if isinstance(item.content, str):
                item.content = [item.content]
            for i, content in enumerate(item.content):
                self._column_widths[i] = max(
                    self._column_widths.get(i, 0), len(str(content))
                )
        if headers:
            self._headers = headers
            for i, header in enumerate(headers):
                self._column_widths[i] = max(self._column_widths.get(i, 0), len(header))

        # get column sizing if specified
        self._column_sizing = kwargs.pop("column_sizing", None)
        # get column alignment if specified
        self._column_align = kwargs.pop("column_align", None)

        for item in items:
            item.column_widths = self._column_widths
            if attr_map:
                item.column_attr_map = attr_map
            if focus_map:
                item.column_focus_map = focus_map
            item._column_sizing = self._column_sizing
            item._column_align = self._column_align
            item._invalidate()

        self._items = items
        self._always_in_focus = always_in_focus
        self._listwalker = urwid.SimpleFocusListWalker(self.items)
        super().__init__(self._listwalker, *args, **kwargs, keymap_jk=True)

    @property
    def items(self) -> List[cActionItem]:
        return self._items

    @items.setter
    def items(self, value: List[cActionItem]) -> None:
        self._items = value
        self.body[:] = value

    def keypress(self, size: tuple[int], key: str) -> Union[str, None]:
        return super().keypress(size, key)

    def render(self, size: tuple[int], focus: bool) -> Tuple[int, int]:
        if self._always_in_focus:
            focus = True
        return super().render(size, focus)


class cPopup(urwid.Overlay):
    """A custom Popup widget with a title, frame and footer.

    Args:
        - widget (urwid.Widget): The main widget to display in the popup.
        - *args: Additional positional arguments.
        - **kwargs: Additional keyword arguments.

    Keyword Arguments:
        - height (int): The height of the popup.
        - footer (str): The text to display in the footer.
        - align (str): The alignment of the popup.
        - width (tuple[str, int]): The width of the popup.
        - valign (str): The vertical alignment of the popup.
        - min_height (int): The minimum height of the popup.
        - title (str): The title of the popup.
        - title_align (str): The alignment of the title.
        - focus_map (str): The focus map for the popup.
    """

    def __init__(self, widget: urwid.Widget, *args, **kwargs):
        if "height" in kwargs:
            if isinstance(kwargs["height"], int):
                kwargs["height"] += 4

        kwargs.setdefault("footer", "Enter: Open | Esc: Close")
        kwargs.setdefault("align", "center")
        kwargs.setdefault("width", ("relative", 80))
        kwargs.setdefault("valign", "middle")
        kwargs.setdefault("height", ("relative", 50))
        kwargs.setdefault("min_height", 5)
        kwargs.setdefault("title", "")
        kwargs.setdefault("title_align", "left")
        kwargs.setdefault("focus_map", "8")

        self._widget = widget
        self._frame = urwid.Frame(
            urwid.Padding(widget, left=1, right=1),
            footer=urwid.Pile(
                [
                    urwid.Divider("\u2500"),
                    urwid.Padding(
                        urwid.Text(("2", kwargs.pop("footer"))), left=1, right=1
                    ),
                ]
            ),
            focus_part="body",
        )

        box = cRoundedLineBox(
            self._frame,
            title=kwargs.pop("title"),
            title_align=kwargs.pop("title_align"),
            focus_map=kwargs.pop("focus_map"),
        )

        base_widget = (
            Caller.main_loop.widget if Caller.main_loop else urwid.SolidFill(" ")
        )

        super().__init__(box, base_widget, *args, **kwargs)

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        return super().keypress((1,), key)
