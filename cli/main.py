import argparse

import pandas as pd
import rich

from bamboost import _config, index
from bamboost.index import DatabaseTable, IndexAPI
from bamboost.manager import Manager
from bamboost.simulation import Simulation
from bamboost.simulation_writer import SimulationWriter


def main():
    parser = argparse.ArgumentParser(description="CLI for bamboost")
    parser.add_argument(
        "--remote", "-r", help="Remote server to fetch data from", type=str
    )
    subparsers = parser.add_subparsers(
        dest="command",
    )
    function_map = {}

    # ----------------
    # Submit
    # ----------------
    function_map["submit"] = submit_simulation
    parser_submit = subparsers.add_parser("submit", help="Submit a job")
    parser_submit.add_argument(
        "path",
        nargs="?",
        type=str,
        help="Path to simulation (directory) to submit",
    )
    parser_submit.add_argument(
        "--id",
        default=None,
        type=str,
        help="Simulation ID",
    )
    parser_submit.add_argument(
        "--db",
        default=None,
        type=str,
        help="Database ID in which a submission will be made",
    )
    parser_submit.add_argument(
        "--all",
        action="store_true",
        help="Submit all unsubmitted simulations in the database",
    )

    # ----------------
    # Database
    # ----------------
    function_map["db"] = manage_db
    parser_db = subparsers.add_parser("db", help="Database management")
    parser_db.add_argument("db_id", type=str, help="Database ID")
    sub_parser_db = parser_db.add_subparsers(
        dest="subcommand",
        help="Database management subcommand",
    )
    sub_parser_db.add_parser("list", help="List all simulations in the database")
    sub_parser_db.add_parser("drop", help="Drop database from the index")
    sub_parser_db.add_parser("reset", help="Reset its table in the sqlite database")

    # ----------------
    # List
    # ----------------
    function_map["list"] = list_databases
    parser_list = subparsers.add_parser("list", help="List all databases")  # noqa: F841

    # ----------------
    # Clean
    # ----------------
    function_map["clean"] = clean_index
    parser_clean = subparsers.add_parser("clean", help="Clean the index")
    parser_clean.add_argument(
        "--purge",
        action="store_true",
        help="also remove the tables from unknown databases",
    )

    # ----------------
    # Scan
    # ----------------
    def scan_known_paths(args):
        IndexAPI().scan_known_paths()
        rich.print("Scanned known paths")

    function_map["scan"] = scan_known_paths

    parser_scan = subparsers.add_parser("scan", help="Scan known paths")  # noqa: F841

    # ----------------
    # Open config file
    # ----------------
    function_map["config"] = open_config
    parser_config = subparsers.add_parser("config", help="Open the config file")
    parser_config.add_argument(
        "--tui", "-t", action="store_true", help="Open the tui config file"
    )
    parser_config.add_argument(
        "--functions", "-f", action="store_true", help="Open the tui functions file"
    )

    # ----------------
    # New database
    # ----------------
    function_map["new"] = create_new_database
    parser_new = subparsers.add_parser("new", help="Create a new database")
    parser_new.add_argument("path", type=str, help="Path to the database")

    # ----------------
    # Parse
    # ----------------
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    func = function_map[args.command]
    func(args)


def submit_simulation(args):
    if args.path is not None:
        db_path, uid = args.path.rstrip("/").rsplit("/", 1)
        sim = Simulation(uid, db_path)
        args.id = uid
        args.db = IndexAPI().get_id(db_path)
    else:
        assert args.id and args.db, "Simulation ID and database ID must be provided"
        sim = Simulation(args.id, args.db)

    sim.submit()
    print(f"Submitted simulation {args.id} [db: {args.db}]")


def manage_db(args):
    # test if the database exists
    df = IndexAPI().read_table()
    if args.db_id not in df["id"].values:
        match = df[df["path"].astype(str).str.contains(args.db_id, na=False)].reset_index(drop=True)
        choice = 0
        if len(match) > 1:
            rich.print(
                f"Database ID {args.db_id} is ambiguous. Possible matches: \n{match.to_string()}"
            )
            choice = int(input("Choose one of the above databases to continue: "))
        args.db_id = match.iloc[choice]["id"]

    if args.subcommand == "list":
        rich.print(Manager(uid=args.db_id).df)
    if args.subcommand == "reset":
        with IndexAPI().open():
            IndexAPI()._conn.execute(f"DROP TABLE IF EXISTS db_{args.db_id}")
            IndexAPI()._conn.execute(f"DROP TABLE IF EXISTS db_{args.db_id}_t")
        rich.print(f"Database {args.db_id} dropped")

    if args.subcommand == "drop":
        with IndexAPI().open():
            IndexAPI()._conn.execute(f"DROP TABLE IF EXISTS db_{args.db_id}")
            IndexAPI()._conn.execute(f"DROP TABLE IF EXISTS db_{args.db_id}_t")
        IndexAPI().drop_path(args.db_id)
        rich.print(f"Database {args.db_id} dropped")


def list_databases(args):
    table = IndexAPI().read_table()
    rich.print(table.to_string())


def clean_index(args, purge: bool = False):
    IndexAPI().clean(purge=purge)
    rich.print("Index cleaned")


def open_config(args):
    import os
    import subprocess

    if args.tui:
        subprocess.run(
            [
                f"{os.environ.get('EDITOR', 'vi')}",
                f"{_config.paths['CONFIG_DIR']}/tui.toml",
            ]
        )
    elif args.functions:
        subprocess.run(
            [
                f"{os.environ.get('EDITOR', 'vi')}",
                f"{_config.paths['CONFIG_DIR']}/custom_functions.py",
            ]
        )
    else:
        subprocess.run(
            [f"{os.environ.get('EDITOR', 'vi')}", f"{_config.paths['CONFIG_FILE']}"]
        )


def create_new_database(args):
    try:
        path = index.get_uid_from_path(args.path)
        rich.print(f"Database at {args.path} already exists. ID: {path}")
    except FileNotFoundError:
        db = Manager(args.path, create_if_not_exist=True)
        rich.print(f"Database at {args.path} created. ID: {db.UID}")
