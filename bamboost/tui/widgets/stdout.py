import urwid
from bamboost.tui.widgets.custom_widgets import cPopup


class StdoutWidget(cPopup):
    """A widget to display the output of third party commands."""

    class TextWithWrite(urwid.Text):
        def write(self, data: str) -> None:
            self.set_text(self.text + data)

        def flush(self, *args) -> None:
            pass

    def __init__(self) -> None:
        super().__init__(
            urwid.Filler(StdoutWidget.TextWithWrite(""), valign="top"),
            title="stdout",
            height=("relative", 50),
            footer="Press 'q' to close",
            valign="bottom",
            width=("relative", 100),
        )
