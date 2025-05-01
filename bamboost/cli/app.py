from pathlib import Path
from typing import Optional

import rich
import typer
from typing_extensions import Annotated

import bamboost.cli.database as database_cli
from bamboost.cli import _completion, _render
from bamboost.cli._fast_index_query import INDEX

app = typer.Typer(no_args_is_help=True)
app.add_typer(database_cli.app)
app.registered_commands.extend(database_cli.app.registered_commands)
console = rich.get_console()


@app.callback(context_settings={"help_option_names": ["-h", "--help"]})
def _app_callback(ctx: typer.Context):
    """Command line interface for bamboost.

    You can use this to list collections or simulations, create collections/simulations,
    and run simulations.
    """


@database_cli.app.command("ls")
def list_collections():
    """List all collections."""
    _assert_database_exists()
    console.print(_render._list_collections())


@app.command()
@app.command("ls", hidden=True)
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
):
    """List all collections, or all simulations in a specified collection. (alias: "ls")"""
    _assert_database_exists()

    # If the user has not provided a collection UID, list all collections
    if collection_uid is None:
        return console.print(_render._list_collections())

    with console.status("[bold blue]Fetching data...", spinner="dots"):
        # if the user has provided a collection UID, list all simulations in that collection
        # specifically handle the case where the collection is empty
        if simulation_name is None:
            try:
                df = _render._list_simulations(collection_uid, sync_fs)
            except ValueError as e:
                return console.print(str(e), style="red")

            if df.empty:
                return console.print(
                    f"Collection [bold]{collection_uid}[/bold] is empty.\n"
                    "[dim]If you expected something here: Use '-s' to sync the collection with the filesystem.[/dim]"
                )

            collection_path = INDEX.query(
                "SELECT path FROM collections WHERE uid = ?", (collection_uid,)
            )[0][0]
            return console.print(
                f"[dim]> Displaying collection [bold]{collection_uid}[/bold] "
                f"([default]{collection_path}[/default]):\n",
                df,
            )

        # if the user has provided a simulation name, display the simulation details
        else:
            from bamboost.index import Index

            sim = Index.default.simulation(collection_uid, simulation_name)
            if sim is None:
                return console.print(
                    f"Simulation [bold]{simulation_name}[/bold] not found in collection [bold]{collection_uid}[/bold].",
                    style="red",
                )
            return console.print(sim.as_dict())


@app.command()
def show_config(
    dir: Optional[str] = typer.Option(
        None, "--dir", "-d", help="Directory to show the config for."
    ),
):
    """Show the active configuration."""
    if dir:
        from bamboost._config import _Config

        config = _Config(dir)
    else:
        from bamboost import config

    console.print(config)


@app.command()
def new(
    path: Annotated[str, typer.Argument(..., help="Path of the collection to create.")],
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
        console.print("[green]:heavy_check_mark: New collection created")
        console.print(f"[default]{'UID:':<5} {coll.uid}")
        console.print(f"[default]{'Path:':<5} {coll.path}")


def _assert_database_exists() -> bool:
    """Check if the database exists."""
    from bamboost._config import config

    if config.index.databaseFile.exists():
        return True

    console.print(f"Creating database file {config.index.databaseFile}")
    database_cli.scan()
    return True
