from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from simple_parsing import (
    ArgumentGenerationMode,
    ArgumentParser,
    NestedMode,
    field,
)

from bamboost.cli._collection import Collection
from bamboost.cli._formatter import Formatter
from bamboost.cli._index import Index


@dataclass
class Submit:
    """Submit a simulation"""

    def execute(self):
        pass


@dataclass
class BamboostCli:
    """The command line interface of bamboost."""

    subcommand: Union[Collection, Submit, Index]
    remote: Optional[str] = field(
        alias="-r", help="The remote ssh host to interact with.", default=None
    )

    def execute(self):
        self.subcommand._parser = self._parser
        self.subcommand.execute()


def main():
    parser = ArgumentParser(formatter_class=Formatter)
    parser.add_arguments(BamboostCli, "bamboost")

    args = parser.parse_args()
    cli: BamboostCli = args.bamboost
    cli._parser = parser
    cli.execute()


if __name__ == "__main__":
    main()
