# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

"""Utility functions used by bamboost.
"""

from collections.abc import MutableMapping
from itertools import islice
from pathlib import Path

import h5py

__all__ = ["flatten_dict", "unflatten_dict", "tree", "h5_tree"]

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
    folder_symbol = "\U00002B57 "

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
                yield prefix + pointer + "\U000025CC " + path.name
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
