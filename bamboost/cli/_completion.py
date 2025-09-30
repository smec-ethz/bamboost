import typer

from bamboost.cli._fast_index_query import INDEX


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


def _get_aliases_of_collection(ctx: typer.Context, incomplete: str):
    """Dynamically fetch aliases based on the selected collection UID."""
    collection_uid = ctx.params.get("uid")  # Get the currently selected collection
    if not collection_uid:
        return []  # No collection selected yet, no autocompletion

    try:
        aliases = INDEX.query(
            "SELECT aliases FROM collections WHERE uid = ?", (collection_uid,)
        )
        if aliases:
            import json

            return [
                alias
                for alias in json.loads(aliases[0][0])
                if alias.startswith(incomplete)
            ]
        return []
    except Exception:
        return []
