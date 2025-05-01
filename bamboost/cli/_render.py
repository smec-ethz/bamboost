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


def _list_simulations(collection_uid: str, sync: bool = False) -> "DataFrame":
    from datetime import datetime

    from bamboost.index import Index

    if sync:
        Index.default.sync_collection(collection_uid)

    coll = Index.default.collection(collection_uid)
    if coll is None:
        raise ValueError(f"Collection with UID {collection_uid} not found.")
    tab = coll.to_pandas()

    # Format datetime columns
    for col in tab.select_dtypes(include=["datetime64"]):
        tab[col] = tab[col].apply(lambda x: datetime.strftime(x, "%Y-%m-%d %H:%M:%S"))

    return tab
