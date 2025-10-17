import json
from typing import TYPE_CHECKING, Iterable

from bamboost.cli._fast_index_query import INDEX

if TYPE_CHECKING:
    from pandas import DataFrame
    from rich.console import RenderableType
    from rich.table import Table


def _get_collections_table(collections: Iterable[tuple]) -> "Table":
    from rich.table import Column, Table

    tab = Table(
        "",
        "UID",
        Column("Path", style="blue"),
        Column("Aliases", style="green"),
        Column("Tags", style="magenta"),
        title_justify="left",
        highlight=True,
        pad_edge=False,
        box=None,
    )

    for i, coll in enumerate(collections):
        tab.add_row(
            str(i),
            coll[0],
            f"[link={coll[1].as_uri()}]{coll[1].as_posix()}[/link]",
            ", ".join(coll[2]),
            ", ".join(coll[3]),
        )

    return tab


def _list_collections() -> "RenderableType":
    from pathlib import Path

    tab = _get_collections_table(
        (i, Path(j), json.loads(a), json.loads(t))
        for i, j, a, t in INDEX.query(
            "SELECT uid, path, aliases, tags FROM collections"
        )
    )
    return tab


def _list_simulations(coll, *, nb_entries: int | None = None) -> "DataFrame":
    from datetime import datetime

    tab = coll.to_pandas()
    if nb_entries is not None:
        tab = tab.head(nb_entries)

    # Format datetime columns
    for col in tab.select_dtypes(include=["datetime64"]):
        tab[col] = tab[col].apply(lambda x: datetime.strftime(x, "%Y-%m-%d %H:%M:%S"))

    return tab
