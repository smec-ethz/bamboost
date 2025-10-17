from __future__ import annotations

from typing import Annotated, Optional

import typer

from bamboost.cli import _completion, _render
from bamboost.cli.common import console, task_status

app_index = typer.Typer(
    name="index",
    help="API for displaying and managing the collection index.",
    no_args_is_help=True,
)


def _assert_database_exists() -> bool:
    """Check if the database exists."""
    from bamboost._config import config

    if config.index.databaseFile.exists():
        return True

    console.print(f"Creating database file {config.index.databaseFile}")
    scan()
    return True


@app_index.callback()
def _app_callback():
    """Display the database file (project path)."""
    from bamboost import config

    console.print(f"[dim default][bold]Index file: [/bold]{config.index.databaseFile}")
    if config.index.isolated:
        console.print(f"[dim default][bold]Project: [/bold]{config.index.projectDir}")


@app_index.command()
@app_index.command("ls", hidden=True)
def list(
    collection_uid: Annotated[
        Optional[str],
        typer.Argument(
            autocompletion=_completion._get_uids_from_db,
            help="UID of the collection to list simulations for.",
        ),
    ] = None,
    simulation_name: Annotated[
        Optional[str],
        typer.Argument(
            help="Name of the simulation to display. If not provided, the collection is listed.",
            autocompletion=_completion._get_simulation_names,
        ),
    ] = None,
    sync_fs: bool = typer.Option(
        False, "--sync", "-s", help="Sync the collection with the filesystem."
    ),
    nb_entries: Optional[int] = typer.Option(
        None, "--entries", "-n", help="Number of entries to display."
    ),
):
    """List all collections, or all simulations in a specified collection. (alias: "ls")"""
    _assert_database_exists()

    # If the user has not provided a collection UID, list all collections
    if collection_uid is None:
        return console.print(_render._list_collections())

    with console.status("[bold blue]Fetching data...", spinner="dots"):
        # if the user has provided a simulation name, display the simulation details
        if simulation_name:
            from bamboost.index import Index

            sim = Index.default.simulation(collection_uid, simulation_name)
            if sim is None:
                return console.print(
                    f"Simulation [bold]{simulation_name}[/bold] not found in collection [bold]{collection_uid}[/bold].",
                    style="red",
                )
            return console.print(sim.as_dict())

        # if the user has provided a collection UID, list all simulations in that collection
        # specifically handle the case where the collection is empty
        from bamboost.index import Index

        with Index.default.sql_transaction():
            coll_uid = Index.default._find_collection_uid_by_alias(collection_uid)
            if coll_uid is None:
                raise ValueError(f"Collection with UID {collection_uid} not found.")

        if sync_fs:
            console.print(f"Syncing collection [bold]{collection_uid}[/bold]...")
            Index.default.sync_collection(coll_uid)

        coll = Index.default.collection(coll_uid)

        try:
            df = _render._list_simulations(coll, nb_entries=nb_entries)
        except ValueError as e:
            return console.print(str(e), style="red")

        if df.empty:
            return console.print(
                f"Collection [bold]{coll.uid}[/bold] is empty.\n"
                "[dim]If you expected something here: Use '-s' to sync the collection with the filesystem.[/dim]"
            )

        return console.print(
            f"[dim]> Displaying collection [bold]{coll.uid}[/bold] "
            f"([default]{coll.path}[/default]):\n",
            df,
        )


@app_index.command()
def clean():
    """Clean the index of any stale entries."""
    with task_status("[bold blue]Cleaning index...", "[green]Index cleaned."):
        from bamboost.index import Index

        Index.default.check_integrity()


@app_index.command()
def scan():
    """Scan the search paths for collections."""
    with task_status(
        "[bold blue]Scanning search paths...",
        "Index scanned.",
    ):
        from bamboost.index import Index

        Index.default.scan_for_collections()


@app_index.command()
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
        from bamboost.index import Index

        Index.default._drop_collection(uid)
        console.print(
            f"[green]:heavy_check_mark: Collection '{uid}' dropped from index."
        )
