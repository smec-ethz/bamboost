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
    import sqlite3

    from bamboost import config
    from bamboost.index import Index

    # db name of version 0.10.x
    db_name_0_10 = "bamboost-next.sqlite"

    db_file_0_10 = Path(config.paths.localDir).joinpath(db_name_0_10)

    if not db_file_0_10.exists():
        return console.print(
            f"[red]:cross_mark: Nothing to migrate. Database file for version 0.10.x not found at {db_file_0_10}"
        )

    console.print("[yellow]🔄 Starting migration from 0.10.x → 0.11.x...")

    # make sure the new database file exists by initializing an Index instance
    Index()
    # after, we use manual sqlite3 commands to copy data
    new_db_file = config.index.databaseFile

    # from 0.10 to 0.11 here's what we have to do:
    # - collection table: read uid, path -> upsert collections
    # - simulation table: clone simulation table to new database
    # - parameters table: clone parameters table to new database
    # open both DBs
    with (
        sqlite3.connect(db_file_0_10) as conn_old,
        sqlite3.connect(new_db_file) as conn_new,
    ):
        cur_old = conn_old.cursor()
        cur_new = conn_new.cursor()

        # === collections ===
        cur_old.execute("SELECT uid, path FROM collections")
        collections = cur_old.fetchall()
        cur_new.executemany(
            """
            INSERT OR IGNORE INTO collections (uid, path, description, tags, aliases, author)
            VALUES (?, ?, '', '[]', '[]', 'null')
            """,
            [(uid, path) for uid, path in collections],
        )

        # === simulations ===
        cur_old.execute(
            "SELECT id, collection_uid, name, created_at, modified_at, description, status, submitted FROM simulations"
        )
        simulations = cur_old.fetchall()
        cur_new.executemany(
            """
            INSERT OR IGNORE INTO simulations
            (id, collection_uid, name, created_at, modified_at, description, status, submitted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            simulations,
        )

        # === parameters ===
        cur_old.execute("SELECT id, simulation_id, key, value FROM parameters")
        parameters = cur_old.fetchall()
        cur_new.executemany(
            """
            INSERT OR IGNORE INTO parameters (id, simulation_id, key, value)
            VALUES (?, ?, ?, ?)
            """,
            parameters,
        )

        conn_new.commit()

    console.print("[green]✅ Migration completed successfully.")
