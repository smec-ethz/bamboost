# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

"""Utility functions used by bamboost."""

from argparse import ArgumentParser
from collections.abc import MutableMapping
from itertools import islice
from pathlib import Path
from typing import NamedTuple

import h5py
import pandas as pd

__all__ = [
    "flatten_dict",
    "unflatten_dict",
    "tree",
    "h5_tree",
    "show_differences",
    "to_camel_case",
]

space = "    "
branch = "│   "
tee = "├── "
last = "└── "


def flatten_dict(dictionary, parent_key="", seperator="."):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + seperator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, seperator=seperator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(dictionary, seperator="."):
    new_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(seperator)
        d = new_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return new_dict


# https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
def tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
):
    """Given a directory Path object print a visual tree structure"""
    dir_path = Path(dir_path)  # accept string coerceable to Path
    files = 0
    directories = 0
    folder_symbol = "\U00002b57 "

    def inner(dir_path: Path, prefix: str = "", level=-1):
        nonlocal files, directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + "\U000025cc " + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension, level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    tree_string = ""
    tree_string += (folder_symbol + dir_path.name) + "\n"
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        tree_string += (line) + "\n"
    if next(iterator, None):
        tree_string += (f"... length_limit, {length_limit}, reached, counted:") + "\n"
    tree_string += (
        f"\n{directories} directories" + (f", {files} files" if files else "")
    ) + "\n"
    return tree_string


def h5_tree(val, pre=""):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + "└── " + key)
                h5_tree(val, pre + "    ")
            else:
                print(pre + "└── " + key + " (%d)" % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + "├── " + key)
                h5_tree(val, pre + "│   ")
            else:
                print(pre + "├── " + key + " (%d)" % len(val))


def show_differences(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes a pandas DataFrame as input and returns a modified
    DataFrame that shows only the columns which have differences.

    The function first creates a copy of the input DataFrame to work with. It
    then iterates over each column in the DataFrame and tries to calculate the
    number of unique values in that column. If successful, it adds the column
    name and number of unique values to a list of good results. If there is an
    error, it attempts to apply json.dumps to the column and then calculate the
    number of unique values again. If this is successful, it also adds the
    column name and number of unique values to the list of good results. If
    there is still an error, it adds the column name and the error to a list of
    errors.

    After processing all columns, the function removes any columns that had
    errors from the DataFrame. It then sets the index of the DataFrame to 'id'
    and filters out any columns that have only one unique value. The modified
    DataFrame is then returned.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze

    Returns:
        pd.DataFrame
    """
    import json

    df_diff = df.copy()
    cols_nunique_good = []
    cols_nunique_error = []
    for col in df_diff.columns:
        try:
            nunique = df_diff[col].nunique()
            cols_nunique_good.append((col, nunique))
        except Exception:
            try:
                df_diff[col] = df_diff[col].apply(json.dumps)
                nunique = df_diff[col].nunique()
                cols_nunique_good.append((col, nunique))
            except TypeError as e:
                cols_nunique_error.append((col, e))

    df_diff = df_diff[
        df_diff.columns[~df_diff.columns.isin([col for col, _ in cols_nunique_error])]
    ]
    try:
        df_diff.set_index("id", inplace=True)
    except KeyError:
        pass
    df_diff = df_diff.loc[:, (df_diff.nunique() != 1)]
    df_diff.dropna(axis=1, how="all", inplace=True)
    return df_diff


def to_camel_case(s: str) -> str:
    words = s.split()
    camel_case = words[0].lower() + "".join([word.capitalize() for word in words[1:]])
    return camel_case


JobArguments = NamedTuple(
    "JobArguments", [("db_path", Path), ("name", str), ("submit", bool)]
)
ScriptArguments = NamedTuple("ScriptArguments", [("simulation", str)])


def parse_job_arguments() -> JobArguments:
    """Parse command-line arguments for submitting a job to a bamboost database.

    Returns:
        JobArguments: A named tuple containing the parsed arguments.
    """
    parser = ArgumentParser(description="Submit a job to a bamboost database")

    # Add the database path argument
    parser.add_argument("db_path", type=Path, help="Path to the bamboost database")
    # Add the name argument as optional
    parser.add_argument("name", type=str, nargs="?", help="Name of the simulation")
    # Add the submit flag as optional
    parser.add_argument("--submit", "-s", help="Submit the job", action="store_true")

    args = parser.parse_args()

    return JobArguments(db_path=args.db_path, name=args.name, submit=args.submit)


def parse_script_arguments() -> ScriptArguments:
    """Parse command-line arguments for a script using the bamboost system.

    Returns:
        ScriptArguments: A named tuple containing the parsed arguments.
    """
    parser = ArgumentParser(description="Submit a job to a bamboost database")

    # Add the simulation UID argument
    parser.add_argument(
        "--simulation", type=str, help="UID of the simulation", required=True
    )

    args = parser.parse_args()

    return ScriptArguments(simulation=args.simulation)
