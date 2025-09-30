from __future__ import annotations

import rich
import typer

from bamboost.cli import _completion
from bamboost.cli.common import task_status

app = typer.Typer(
    name="index",
    help="API for displaying and managing the collection index.",
    no_args_is_help=True,
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
    with task_status("[bold blue]Cleaning index...", "[green]Index cleaned."):
        from bamboost.index import Index

        Index.default.check_integrity()


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


@app.command()
def drop(
    uid: str = typer.Argument(
        ...,
        autocompletion=_completion._get_uids_from_db,
        help="The unique ID of the collection to drop from the index.",
    ),
) -> None:
    """Drop a collection from the index by its unique ID (or an alias)."""
    with console.status(
        f"[bold blue]Dropping collection '{uid}' from index...", spinner="dots"
    ) as status:
        try:
            from bamboost.index import Index

            Index.default._drop_collection(uid)
            console.print(
                f"[green]:heavy_check_mark: Collection '{uid}' dropped from index."
            )
        except Exception as e:
            console.print(f"[bold red]Task failed: {e}")


subapp_alias = typer.Typer(name="alias", help="Manage collection aliases in the index.")
app.add_typer(subapp_alias)


@subapp_alias.command()
def add(
    uid: str = typer.Argument(
        ...,
        autocompletion=_completion._get_uids_from_db,
        help="The unique ID of the collection to add an alias for.",
    ),
    alias: str = typer.Argument(..., help="The alias to add to the collection."),
) -> None:
    """Add an alias to a collection in the index by its unique ID."""
    with console.status(
        f"[bold blue]Adding alias '{alias}' to collection '{uid}'...", spinner="dots"
    ) as status:
        try:
            from pathlib import Path

            from bamboost.index import Index

            coll_record = Index.default.collection(uid)
            Index.default.upsert_collection(
                coll_record.uid,
                Path(coll_record.path),
                {"aliases": coll_record.aliases + [alias]},
            )
            console.print(
                f"[green]:heavy_check_mark: Alias '{alias}' added to collection '{uid}'."
            )
        except Exception as e:
            console.print(f"[bold red]Task failed: {e}")


@subapp_alias.command()
def remove(
    uid: str = typer.Argument(
        ...,
        autocompletion=_completion._get_uids_from_db,
        help="The unique ID of the collection to remove an alias from.",
    ),
    alias: str = typer.Argument(
        ...,
        help="The alias to remove from the collection.",
        autocompletion=_completion._get_aliases_of_collection,
    ),
) -> None:
    """Remove an alias from a collection in the index by its unique ID."""
    with console.status(
        f"[bold blue]Removing alias '{alias}' from collection '{uid}'...",
        spinner="dots",
    ) as status:
        try:
            from pathlib import Path

            from bamboost.index import Index

            coll_record = Index.default.collection(uid)
            if alias not in coll_record.aliases:
                raise ValueError(f"Alias '{alias}' not found in collection '{uid}'.")
            new_aliases = [a for a in coll_record.aliases if a != alias]
            Index.default.upsert_collection(
                coll_record.uid,
                Path(coll_record.path),
                {"aliases": new_aliases},
            )
            console.print(
                f"[green]:heavy_check_mark: Alias '{alias}' removed from collection '{uid}'."
            )
        except Exception as e:
            console.print(f"[bold red]Task failed: {e}")
