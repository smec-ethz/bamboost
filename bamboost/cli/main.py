from __future__ import annotations

import argparse
from functools import wraps
from typing import Callable, List, Optional, Type, Union


class Argument:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, parser: argparse.ArgumentParser):
        parser.add_argument(*self.args, **self.kwargs)


class Cli:
    parser: argparse.ArgumentParser
    subcli: List[Type[Cli]]
    aliases: List[str]
    execute: Callable[[], None]

    def __init__(self, parser: Optional[argparse.ArgumentParser] = None):
        self.parser = parser or argparse.ArgumentParser(description=self.__doc__)
        self.__subclis: dict[str, Cli] = {}

        # add the options
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, Argument):
                value(self.parser)

        # add subparsers
        if hasattr(self, "subcli"):
            subparsers = self.parser.add_subparsers(dest="subcli")
            for cli in self.subcli:
                subparser = subparsers.add_parser(
                    cli.__name__.lower(),
                    help=cli.__doc__,
                    description=cli.__doc__,
                    aliases=cli.aliases,
                )
                # init the subparser
                self.__subclis[cli.__name__.lower()] = cli(subparser)

    def parse(self) -> Cli:
        args = self.parser.parse_args()

        def get_final_subcli(cli: Cli, args) -> Cli:
            if hasattr(cli, "subcli") and args.subcli:
                for i in cli.subcli:
                    if args.subcli == i.__name__.lower():
                        subcli_selected = cli.__subclis[args.subcli]
                        for attr, value in args.__dict__.items():
                            if attr == 'subcli':
                                continue
                            setattr(subcli_selected, attr, value)
                        return get_final_subcli(subcli_selected, args)
            return cli

        return get_final_subcli(self, args)


class CliParsed:
    subcli: Type[Cli]


class Submit(Cli):
    """Run jobs or submit jobs on a cluster using slurm."""

    aliases = []
    # fmt: off
    path = Argument("path", nargs="?", type=str, help="Path to the directory containing the simulation.")
    id = Argument("--id", default=None, type=str, help="The id of the simulation.")
    db = Argument("--db", default=None, type=str, help="The ID of the collection containing the simulation.")
    all = Argument("--all", action="store_true", help="Submit all unsubmitted jobs in the Collection.")

    # fmt: on
    def execute(self):
        print("Submit")


class Collection(Cli):
    """Collection management commands."""

    aliases = ["coll"]

    def execute(self):
        print("Db")


class MainCli(Cli):
    """The command line interface for bamboost."""

    subcli = [Submit, Collection]
    """The subcommands of the main CLI."""

    remote = Argument(
        "-r", "--remote", help="Remote ssh database to fetch data from.", type=str
    )


def main():
    cli = MainCli()
    a = cli.parse()
    print(a)
    print(a.__dict__)


if __name__ == "__main__":
    main()
