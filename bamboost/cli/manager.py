import argparse
from dataclasses import dataclass
from typing import Literal, Optional, Union

import rich
import rich_argparse
from simple_parsing import (
    ArgumentParser,
    field,
)

from bamboost.cli._formatter import Formatter


@dataclass
class Submit:
    """Submit a simulation"""

    def execute(self):
        pass


@dataclass
class Collection:
    """Display or manage a collection."""

    def execute(self):
        pass


@dataclass
class Index:
    """Show and interact with the index."""

    list: str = None
    """List all collections in the index."""
    uid: str = field(
        alias="-u", default=None, help="The uid of the collection to interact with."
    )

    def execute(self):
        import rich.box
        from rich.table import Table

        from bamboost.index import DEFAULT_INDEX

        tab = Table(
            "",
            "UID",
            "Path",
            title="Showing all collections in the index",
            title_justify="left",
            highlight=True,
            box=None,
            expand=True,
        )

        for i, coll in enumerate(DEFAULT_INDEX.all_collections):
            tab.add_row(str(i), coll.uid, coll.path)

        rich.print(tab)


@dataclass
class BamboostCli:
    """The command line interface of bamboost."""

    subcommand: Union[Collection, Submit, Index]
    remote: Optional[str] = field(
        alias="-r", help="The remote ssh host to interact with.", default=None
    )

    def execute(self):
        self.subcommand.execute()


def main():
    parser = ArgumentParser(formatter_class=Formatter)
    parser.add_arguments(BamboostCli, "bamboost")

    args = parser.parse_args()
    cli: BamboostCli = args.bamboost
    cli.execute()


if __name__ == "__main__":
    main()
