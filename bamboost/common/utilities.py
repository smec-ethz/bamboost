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


def flatten_dict(dictionary, parent_key='', seperator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + seperator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, seperator=seperator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def unflatten_dict(dictionary, seperator='.'):
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
