import urwid


class cRoundedLineBox(urwid.AttrWrap):
    def __init__(self, *args, attr_map=None, focus_map=None, **kwargs):
        self.widget = urwid.LineBox(
            *args,
            tlcorner=urwid.LineBox.Symbols.LIGHT.TOP_LEFT_ROUNDED,
            trcorner=urwid.LineBox.Symbols.LIGHT.TOP_RIGHT_ROUNDED,
            blcorner=urwid.LineBox.Symbols.LIGHT.BOTTOM_LEFT_ROUNDED,
            brcorner=urwid.LineBox.Symbols.LIGHT.BOTTOM_RIGHT_ROUNDED,
            **kwargs,
        )
        if attr_map is None:
            attr_map = "default"
        if focus_map is None:
            focus_map = "green_box"
        super().__init__(self.widget, attr_map, focus_map)

    def __getattr__(self, name: str):
        return self.widget.base_widget.__getattribute__(name)

    def keypress(self, size: tuple[()], key: str) -> str:
        return super().keypress(size, key)
