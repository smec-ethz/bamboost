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


def _get_collection_sim_completion(ctx: typer.Context, incomplete: str):
    """Dynamically autocomplete collection_uid:simulation_name."""
    import json

    # 1. First step: Suggest the collection UID or alias (append a colon)
    if ":" not in incomplete:
        rows = INDEX.query("SELECT uid, path, aliases FROM collections")
        results = []
        for uid, path, aliases_json in rows:
            if uid.startswith(incomplete):
                results.append((f"{uid}:", path))

            if aliases_json:
                try:
                    aliases = json.loads(aliases_json)
                    for alias in aliases:
                        if alias.startswith(incomplete):
                            results.append((f"{alias}:", f"Alias for {uid} ({path})"))
                except Exception:
                    pass
        return results

    # 2. Second step: Suggest simulations within the selected collection
    else:
        prefix, incomplete_sim = incomplete.split(":", 1)

        # Resolve prefix (which might be a collection UID or an alias) to the real UID
        real_uid = None
        rows = INDEX.query("SELECT uid FROM collections WHERE uid = ?", (prefix,))
        if rows:
            real_uid = prefix
        else:
            # Scan aliases to resolve the real collection UID
            all_cols = INDEX.query("SELECT uid, aliases FROM collections")
            for uid, aliases_json in all_cols:
                if aliases_json:
                    try:
                        aliases = json.loads(aliases_json)
                        if prefix in aliases:
                            real_uid = uid
                            break
                    except Exception:
                        pass

        if not real_uid:
            return []

        # Query simulations belonging to this collection
        names = INDEX.query(
            "SELECT name FROM simulations WHERE collection_uid = ? AND name LIKE ?",
            (real_uid, f"{incomplete_sim}%"),
        )

        return [(f"{prefix}:{name[0]}", f"Simulation in {prefix}") for name in names]
