from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from simple_parsing import ArgumentParser, field


def _get_id_interactively(inline_padding: int = 0) -> str:
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable

    class SelectionApp(App):
        CSS_PATH = "table.tcss"
        ENABLE_COMMAND_PALETTE = False
        INLINE_PADDING = inline_padding

        class Selection(DataTable):
            BINDINGS = [
                ("j", "cursor_down", "Down"),
                ("k", "cursor_up", "Up"),
                ("q", "exit", "Exit"),
            ]
            BORDER_TITLE = "Select a collection"

        def compose(self) -> ComposeResult:
            table = SelectionApp.Selection(
                cursor_type="row", id="selection-table", show_header=False
            )
            yield table

        def on_mount(self) -> None:
            table = self.query_one(DataTable)
            table.add_column("", key="index")
            table.add_column("UID", key="uid")
            table.add_column("Path", key="path")

            from bamboost.index import DEFAULT_INDEX

            for i, coll in enumerate(DEFAULT_INDEX.all_collections):
                table.add_row(i, coll.uid, coll.path)
            table.focus()

        async def on_data_table_row_selected(
            self, event: DataTable.RowSelected
        ) -> None:
            table = event.control
            selected_row = table.get_row(event.row_key)
            selected_id = selected_row[1]
            self.exit(selected_id)

    selected_id = SelectionApp(watch_css=True, ansi_color=True).run(inline=True)
    if selected_id is None:
        raise SystemExit(0)
    return selected_id


@dataclass
class Collection:
    """Scan for collections."""

    uid: str = field(
        positional=True,
        default=False,
        help="The UID of the collection to display.",
    )
    interactive: bool = field(
        alias="i",
        default=False,
        help="Interactively select a collection.",
        metadata={"action": "store_true"},
    )

    def execute(self):
        from rich.console import Console

        console = Console()

        if self.interactive:
            self.uid = _get_id_interactively(inline_padding=0)
        else:
            if self.uid == False:
                console.print(
                    "[bold red]Please provide a collection UID or choose the interactive mode."
                )
                raise SystemExit(1)

        with console.status(
            f"[bold blue]Displaying collection '{self.uid}'...", spinner="dots"
        ) as status:
            import rich.box
            from rich.panel import Panel
            from rich.table import Table

            from bamboost.index import DEFAULT_INDEX

            coll = DEFAULT_INDEX.collection(self.uid)

            if not coll:
                console.print(f"[bold red]Collection with UID '{self.uid}' not found.")
                raise SystemExit(1)

            from bamboost.core import Collection as _Collection

            df = _Collection(uid=coll.uid).df

            def render_datetime(dt: datetime) -> str:
                return dt.strftime("%Y-%m-%d %H:%M:%S")

            for col in df.select_dtypes(include=["datetime"]).columns:
                df[col] = df[col].apply(render_datetime)
            df = df.astype(str)

            console.print(f"[b]>[/b] Collection [bold]{coll.uid}")
            console.print(df)
            # tab = Table(
            #     "",
            #     *df.columns,
            #     expand=False,
            #     pad_edge=False,
            #     title_justify="left",
            #     highlight=True,
            #     title_style="bold blue",
            #     box=None,
            # )
            # for i, row in enumerate(df.itertuples(index=False)):
            #     tab.add_row(str(i), *row)
            # console.print(f"[b]>[/b] Collection [bold]{coll.uid}")
            # console.print(tab)
