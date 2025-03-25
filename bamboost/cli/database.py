from __future__ import annotations

import rich
import typer

app = typer.Typer(
    name="index", help="API for displaying and managing the collection index."
)
console = rich.get_console()


@app.callback()
def _app_callback():
    """Display the database file (project path)."""
    from bamboost import config

    console.print(f"[dim default][bold]Index file: [/bold]{config.index.databaseFile}")
    if config.index.isolated:
        console.print(f"[dim default][bold]Project: [/bold]{config.index.projectDir}")


@app.command()
def clean():
    """Clean the index of any stale entries."""
    with console.status("[bold blue]Cleaning index...", spinner="dots") as status:
        try:
            from bamboost.index import Index

            Index.default.check_integrity()
            console.print("[green]:heavy_check_mark: Index cleaned.")
        except Exception as e:
            console.print(f"[bold red]Task failed: {e}")


@app.command()
def scan():
    """Scan the search paths for collections."""
    with console.status(
        "[bold blue]Scanning search paths...", spinner="dots"
    ) as status:
        try:
            from bamboost.index import Index

            _found_colls = Index.default.scan_for_collections()
            console.print("[green]:heavy_check_mark: Index scanned.")
            # tab = _get_collections_table(found_colls)
            # console.print(tab)
        except Exception as e:
            console.print(f"[bold red]Task failed: {e}")
