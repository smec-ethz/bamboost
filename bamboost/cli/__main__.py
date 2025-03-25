from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import rich
import typer
from typing_extensions import Annotated

from bamboost.cli._database_api import INDEX
from bamboost.cli.database import app as database

if TYPE_CHECKING:
    from pandas import DataFrame
    from rich.console import RenderableType
    from rich.table import Table


app = typer.Typer(no_args_is_help=True)
app.add_typer(database)
console = rich.get_console()


@app.callback(context_settings={"help_option_names": ["-h", "--help"]})
def _app_callback(ctx: typer.Context):
    """Command line interface for bamboost.

    You can use this to list collections or simulations, create collections/simulations,
    and run simulations.
    """


# Completion function
def _get_uids_from_db(_ctx: typer.Context, incomplete: str):
    """Dynamically fetch UIDs from the database."""
    return [
        row
        for row in INDEX.query(
            "SELECT uid, path FROM collections WHERE uid LIKE ?", (f"{incomplete}%",)
        )
    ]


# Completion function
def _get_simulation_names(ctx: typer.Context, incomplete: str):
    """Dynamically fetch simulation names based on the selected collection UID."""
    collection_uid = ctx.params.get(
        "collection_uid"
    )  # Get the currently selected collection
    if not collection_uid:
        return []  # No collection selected yet, no autocompletion

    try:
        names = INDEX.query(
            "SELECT name FROM simulations WHERE collection_uid = ? AND name LIKE ?",
            (collection_uid, f"{incomplete}%"),
        )
        return [row[0] for row in names]
    except Exception:
        return []


def _get_collections_table(collections: Iterable[tuple]) -> "Table":
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

    for i, coll in enumerate(collections):
        tab.add_row(
            str(i), coll[0], f"[link={coll[1].as_uri()}]{coll[1].as_posix()}[/link]"
        )

    return tab


def _list_collections() -> "RenderableType":
    from pathlib import Path

    tab = _get_collections_table(
        (i, Path(j)) for i, j in INDEX.query("SELECT * FROM collections")
    )
    return tab


def _list_simulations(collection_uid: str) -> "DataFrame":
    from datetime import datetime

    from bamboost.index import Index

    coll = Index.default.collection(collection_uid)
    if coll is None:
        raise ValueError(f"Collection with UID {collection_uid} not found.")
    tab = coll.to_pandas()

    # Format datetime columns
    for col in tab.select_dtypes(include=["datetime64"]):
        tab[col] = tab[col].apply(lambda x: datetime.strftime(x, "%Y-%m-%d %H:%M:%S"))

    return tab


@database.command("ls")
def list_collections():
    """List all collections."""
    console.print(_list_collections())


@app.command()
@app.command("ls", hidden=True)
def list(
    collection_uid: Annotated[
        Optional[str],
        typer.Argument(
            autocompletion=_get_uids_from_db,
            help="UID of the collection to list simulations for.",
        ),
    ] = None,
    simulation_name: Annotated[
        Optional[str],
        typer.Argument(
            help="Name of the simulation to display. If not provided, the collection is listed.",
            autocompletion=_get_simulation_names,
        ),
    ] = None,
):
    """List all collections, or all simulations in a specified collection. (alias: "ls")"""
    # If the user has not provided a collection UID, list all collections
    if collection_uid is None:
        console.print(_list_collections())
    else:
        with console.status("[bold blue]Fetching data...", spinner="dots"):
            # if the user has provided a collection UID, list all simulations in that collection
            # specifically handle the case where the collection is empty
            if simulation_name is None:
                try:
                    df = _list_simulations(collection_uid)
                except ValueError as e:
                    return console.print(str(e), style="red")

                if df.empty:
                    return console.print(
                        f"Collection [bold]{collection_uid}[/bold] is empty. Start generating :)"
                    )

                return console.print(
                    f"[dim]> Displaying collection [bold]{collection_uid}[/bold]:\n", df
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
        console.print(f"[green]:heavy_check_mark: New collection created")
        console.print(f"[default]{'UID:':<5} {coll.uid}")
        console.print(f"[default]{'Path:':<5} {coll.path}")


if __name__ == "__main__":
    app()
