from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from bamboost.cli import _completion, indexing
from bamboost.cli.alias import subapp_alias
from bamboost.cli.common import console
from bamboost.cli.config import app_config

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
app.add_typer(indexing.app_index, rich_help_panel="Subgroups")
app.add_typer(subapp_alias, name="alias", rich_help_panel="Subgroups")
app.add_typer(app_config, name="config", rich_help_panel="Subgroups")

# add from database_cli.app: list, scan, clean
app.command()(indexing.list)
app.command("ls", hidden=True)(indexing.list)
app.command()(indexing.scan)
app.command()(indexing.clean)


@app.callback(context_settings={"help_option_names": ["-h", "--help"]})
def _app_callback(ctx: typer.Context):
    """Command line interface for bamboost.

    You can use this to list collections or simulations, create collections/simulations,
    and run simulations.
    """
    pass


@app.command(no_args_is_help=True)
def new(
    path: Annotated[str, typer.Argument(..., help="Path of the collection to create.")],
    *,
    alias: Annotated[
        Optional[str], typer.Option("--alias", "-a", help="Alias for the collection.")
    ] = None,
    description: Annotated[
        Optional[str],
        typer.Option("--description", "-d", help="Description for the collection."),
    ] = None,
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Comma-separated tags for the collection."),
    ] = None,
) -> None:
    """Create a new collection."""
    path_object = Path(path).resolve()
    if path_object.exists():
        return console.print(f"[red]:cross_mark: Path already in use {path_object}")

    with console.status(
        "[bold blue]Creating new collection...", spinner="dots", spinner_style="blue"
    ):
        from bamboost.core import Collection

        coll = Collection(path_object, create_if_not_exist=True)

        if alias or description or tags:
            from bamboost.index import Index

            Index.default.upsert_collection(
                coll.uid,
                path_object,
                {
                    "aliases": [alias] if alias else [],
                    "description": description or "",
                    "tags": [t.strip() for t in tags.split(",")] if tags else [],
                },
            )

        console.print("[green]:heavy_check_mark: New collection created")
        console.print(f"[default]{'UID:':<5} {coll.uid}")
        console.print(f"[default]{'Path:':<5} {coll.path}")


@app.command()
def yank(
    uid: Annotated[
        str,
        typer.Argument(
            ...,
            help="UID of the collection to yank.",
            autocompletion=_completion._get_uids_from_db,
        ),
    ],
) -> None:
    """Copy the UID of a collection to the clipboard. Only useful with completion."""
    import subprocess

    from bamboost import config

    copy_cmd = config.options.clipboardCommand
    if not copy_cmd:
        return console.print(
            "[red]:cross_mark: Clipboard command not set in config. Please set it first. E.g.\n"
            '\n\toptions.clipboard_command = "wl-copy"  # for Linux with wl-clipboard'
            "\n"
        )
    subprocess.run(copy_cmd, input=uid.encode(), check=True)


@app.command()
def migrate() -> None:
    """Create the new database file from a previous database schema. Useful if you have
    upgraded from 0.10.x to 0.11.x."""
    from bamboost import config
    from bamboost.index.versioning import (
        MigrationError,
        SourceDatabaseNotFoundError,
        Version,
        migrate_database,
    )

    source_db = Path(config.paths.localDir).joinpath(Version.V0_10.database_file_name)
    if not source_db.exists():
        return console.print(
            f"[red]:cross_mark: Nothing to migrate. Database file for version 0.10.x not found at {source_db}"
        )

    console.print("[yellow]🔄 Starting migration from 0.10.x → 0.11.x...")

    try:
        migrate_database(Version.V0_10, Version.V0_11)
    except SourceDatabaseNotFoundError:
        # The file went missing between the initial check and execution.
        console.print(
            f"[red]:cross_mark: Nothing to migrate. Database file for version 0.10.x not found at {source_db}"
        )
    except MigrationError as exc:
        console.print(f"[red]:cross_mark: Migration failed: {exc}")
    else:
        console.print("[green]✅ Migration completed successfully.")
