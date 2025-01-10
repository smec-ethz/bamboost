import time
from dataclasses import dataclass
from typing import Union

from rich.console import Console

from bamboost import config
from bamboost.cli._simple_index import FastIndexQuery

indexAPI = FastIndexQuery(config.index.databaseFile)
"""A fast API relying not using sqlalchemy for simple queries."""

console = Console()


@dataclass
class Show:
    """Show the index."""

    def execute(self):
        import rich
        from rich.table import Column, Table

        tab = Table(
            "",
            "UID",
            Column("Path", style="blue"),
            title_justify="left",
            highlight=True,
            pad_edge=False,
            box=None,
        )

        for i, coll in enumerate(indexAPI.query("SELECT * FROM collections")):
            tab.add_row(str(i), *coll)

        rich.print(tab)


@dataclass
class Clean:
    """Clean the index."""

    def execute(self):
        with console.status("[bold blue]Cleaning index...", spinner="dots") as status:
            try:
                from bamboost.index import DEFAULT_INDEX

                DEFAULT_INDEX.check_integrity()
                console.print("[green]:heavy_check_mark: Index cleaned.")
            except Exception as e:
                console.print(f"[bold red]Task failed: {e}")


@dataclass
class Scan:
    """Scan the search paths for collections."""

    def execute(self):
        with console.status(
            "[bold blue]Scanning search paths...", spinner="dots"
        ) as status:
            try:
                from rich.table import Column, Table

                from bamboost.index import DEFAULT_INDEX

                found_colls = DEFAULT_INDEX.scan_for_collections()
                console.print("[green]:heavy_check_mark: Index scanned.")
                tab = Table(
                    "uid",
                    Column("path", style="blue"),
                    title="Found the following collections:",
                    box=None,
                    show_header=False,
                    pad_edge=False,
                    title_justify="left",
                )
                for coll in found_colls:
                    tab.add_row(
                        coll[0], f"[link={coll[1].as_uri()}]{coll[1].as_posix()}[/link]"
                    )
                console.print(tab)
            except Exception as e:
                console.print(f"[bold red]Task failed: {e}")


@dataclass
class Index:
    """API for displaying and managing the collection index.

    Chose one of the following commands:
        - show: Display the index.
        - clean: Clean the index from invalid collections.
        - scan: Scan the search paths for collections.
    """

    command: Union[Show, Clean, Scan]

    def execute(self):
        self.command.execute()
