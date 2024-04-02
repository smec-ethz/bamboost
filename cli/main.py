import argparse

import rich

from bamboost import _config
from bamboost.index import Index


def main():
    parser = argparse.ArgumentParser(description="CLI for bamboost")
    subparsers = parser.add_subparsers(
        dest="command",
    )

    # Submit
    parser_submit = subparsers.add_parser("submit", help="Submit a job")
    parser_submit.add_argument(
        "path",
        nargs="?",
        type=str,
        help="Database path in which a submission will be made",
    )
    parser_submit.add_argument(
        "--db",
        default=None,
        type=str,
        help="Database ID in which a submission will be made",
    )

    # List
    parser_list = subparsers.add_parser("list", help="List all databases")

    # Clean
    parser_clean = subparsers.add_parser("clean", help="Clean the index")
    parser_clean.add_argument(
        "--purge",
        action="store_true",
        help="also remove the tables from unknown databases",
    )

    # Open config file
    parser_config = subparsers.add_parser("config", help="Open the config file")

    args = parser.parse_args()

    # function map
    functions = {
        "submit": submit_simulation,
        "list": list_databases,
        "clean": clean_index,
        "config": open_config,
    }

    if args.command is None:
        parser.print_help()
        return
    func = functions[args.command]
    func()


def submit_simulation():
    parser = argparse.ArgumentParser()
    print("submitting...")


def list_databases():
    table = Index.read_table()
    rich.print(table.to_string())


def clean_index(purge: bool = False):
    Index.clean(purge=purge)
    rich.print("Index cleaned")


def open_config():
    import subprocess
    import os

    subprocess.run([f"{os.environ.get('EDITOR', 'vi')}", f"{_config.CONFIG_FILE}"])
