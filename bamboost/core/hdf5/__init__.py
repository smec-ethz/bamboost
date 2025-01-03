from typing import Type, Union

import h5py

_VT_filemap = Type[Union[h5py.Group, h5py.Dataset]]

from .file import *
