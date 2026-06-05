import json
from typing import TYPE_CHECKING, Iterable

from bamboost.cli._fast_index_query import INDEX

if TYPE_CHECKING:
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


def _list_simulations(coll, *, nb_entries: int | None = None) -> "Table":
    from datetime import datetime

    from rich.table import Table

    from bamboost.core.utilities import flatten_dict

    simulations = coll.simulations
    if nb_entries is not None:
        simulations = simulations[:nb_entries]

    records = []
    all_keys = set()
    for sim in simulations:
        rec = sim.as_dict(standalone=False, include_links=True)
        rec = flatten_dict(rec)
        records.append(rec)
        all_keys.update(rec.keys())

    # Order keys for columns
    standard_columns = [
        "name",
        "created_at",
        "description",
        "status",
        "submitted",
        "tags",
    ]
    cols = [c for c in standard_columns if c in all_keys]

    param_cols = sorted(
        [
            k
            for k in all_keys
            if k not in standard_columns and not k.startswith("links.")
        ]
    )
    cols.extend(param_cols)

    link_cols = sorted([k for k in all_keys if k.startswith("links.")])
    cols.extend(link_cols)

    tab = Table(
        title_justify="left",
        highlight=True,
        pad_edge=False,
        box=None,
    )

    for col in cols:
        if col == "name":
            tab.add_column("name", style="bold")
        elif col == "created_at":
            tab.add_column("created_at", style="cyan")
        elif col == "status":
            tab.add_column("status", style="magenta")
        elif col.startswith("links."):
            tab.add_column(col, style="blue")
        else:
            tab.add_column(col)

    for rec in records:
        row_vals = []
        for col in cols:
            val = rec.get(col, "")
            if val is None:
                display_val = ""
            elif isinstance(val, datetime):
                display_val = val.strftime("%Y-%m-%d %H:%M:%S")
            elif col == "tags" and isinstance(val, list):
                display_val = ", ".join(val)
            elif col == "submitted":
                display_val = "True" if val else "False"
            else:
                display_val = str(val)
            row_vals.append(display_val)
        tab.add_row(*row_vals)

    return tab
