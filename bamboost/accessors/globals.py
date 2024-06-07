# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import pandas as pd
import numpy as np

from bamboost.common.file_handler import FileHandler
from bamboost.common.hdf_pointer import Group

__all__ = ["GlobalGroup"]


class GlobalGroup(Group):
    """Enhanced Group for '/globals'.

    Args:
        file_handler: The file handler.
        path_to_data: The in-file path to the group.
    """

    def __init__(self, file_handler: FileHandler, path_to_data: str) -> None:
        super().__init__(file_handler, path_to_data)

    @property
    def df(self) -> pd.DataFrame:
        """Return a pandas DataFrame with all datasets."""
        d = {key: self[key][()] for key in self.datasets()}
        for key, val in d.items():
            d[key] = [i for i in val]
        return pd.DataFrame.from_dict(d)
