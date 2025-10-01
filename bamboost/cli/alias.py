from typing import Annotated

import typer

from bamboost.cli import _completion
from bamboost.cli.common import console

subapp_alias = typer.Typer(
    name="alias", help="Manage collection aliases in the index.", no_args_is_help=True
)


@subapp_alias.command()
def add(
    uid: str = typer.Argument(
        ...,
        autocompletion=_completion._get_uids_from_db,
        help="The unique ID of the collection to add an alias for.",
    ),
    alias: str = typer.Argument(..., help="The alias to add to the collection."),
) -> None:
    """Add an alias to a collection in the index by its unique ID."""
    with console.status(
        f"[bold blue]Adding alias '{alias}' to collection '{uid}'...", spinner="dots"
    ) as status:
        try:
            from pathlib import Path

            from bamboost.index import Index

            coll_record = Index.default.collection(uid)
            Index.default.upsert_collection(
                coll_record.uid,
                Path(coll_record.path),
                {"aliases": coll_record.aliases + [alias]},
            )
            console.print(
                f"[green]:heavy_check_mark: Alias '{alias}' added to collection '{uid}'."
            )
        except Exception as e:
            console.print(f"[bold red]Task failed: {e}")


@subapp_alias.command()
def remove(
    uid: str = typer.Argument(
        ...,
        autocompletion=_completion._get_uids_from_db,
        help="The unique ID of the collection to remove an alias from.",
    ),
    alias: str = typer.Argument(
        ...,
        help="The alias to remove from the collection.",
        autocompletion=_completion._get_aliases_of_collection,
    ),
) -> None:
    """Remove an alias from a collection in the index by its unique ID."""
    with console.status(
        f"[bold blue]Removing alias '{alias}' from collection '{uid}'...",
        spinner="dots",
    ) as status:
        from pathlib import Path

        from bamboost.index import Index

        coll_record = Index.default.collection(uid)
        if alias not in coll_record.aliases:
            raise ValueError(f"Alias '{alias}' not found in collection '{uid}'.")
        new_aliases = [a for a in coll_record.aliases if a != alias]
        Index.default.upsert_collection(
            coll_record.uid,
            Path(coll_record.path),
            {"aliases": new_aliases},
        )
        console.print(
            f"[green]:heavy_check_mark: Alias '{alias}' removed from collection '{uid}'."
        )


@subapp_alias.command()
def get(
    alias: Annotated[str, typer.Argument(..., help="The alias to look up.")],
) -> None:
    """Get the unique ID of a collection by its alias."""
    with console.status(
        f"[bold blue]Looking up collection by alias '{alias}'...", spinner="dots"
    ) as status:
        from bamboost.index import Index

        ind = Index.default
        with ind.sql_transaction():
            uid = ind._find_collection_uid_by_alias(alias)
            if uid is None:
                raise ValueError(f"Alias '{alias}' not found in any collection.")
        print(uid)
