"""
This module provides `AttrsDict`, a dictionary-like object that is synced with the
attributes of a group in the HDF5 file.

Key features:
- The object is tied to a simulation. If the simulation is read-only, the object is
  read-only too.
- Automatic synchronization: Updates to the mapping are pushed to the file immediately
  when the simulation is mutable.
- Thread safe and MPI-compatible. Uses the `single_process_queue` of the file object to
  handle attribute updates.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Sequence, Type, Union

import h5py
import numpy as np

from bamboost._typing import _MT, Mutable
from bamboost.core.hdf5.file import (
    FileMode,
    H5Object,
    HDF5File,
    mutable_only,
    with_file_open,
)
from bamboost.core.hdf5.hdf5path import HDF5Path


class _AttrsEncoder:
    def __init__(self):
        self._encoders: Dict[Type, Callable[[Any], Any]] = {}
        self._decoders: Dict[Type, Callable[[Any], Any]] = {}

    def register_encoder(self, typ: Union[Type, Sequence[Type]], encode: Callable):
        if isinstance(typ, Sequence):
            for t in typ:
                self._encoders[t] = encode
        else:
            self._encoders[typ] = encode

    def register_decoder(self, typ: Union[Type, Sequence[Type]], decode: Callable):
        if isinstance(typ, Sequence):
            for t in typ:
                self._decoders[t] = decode
        else:
            self._decoders[typ] = decode

    def encode(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self.encode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.encode(v) for v in obj]
        else:
            for typ, encoder in self._encoders.items():
                if isinstance(obj, typ):
                    encoded = encoder(obj)
                    return json.dumps({"__type__": typ.__name__, "__value__": encoded})
            return obj

    def decode(self, obj: Any) -> Any:
        if isinstance(obj, str):
            try:
                decoded_json = json.loads(obj)
                if (
                    isinstance(decoded_json, dict)
                    and "__type__" in decoded_json
                    and "__value__" in decoded_json
                ):
                    typ_name = decoded_json["__type__"]
                    for typ, decoder in self._decoders.items():
                        if typ.__name__ == typ_name:
                            return decoder(decoded_json["__value__"])
                # fallback to return the original string
                return obj
            except json.JSONDecodeError:
                return obj  # Return the original string if not a special encoded type
        elif isinstance(obj, Mapping):
            return {k: self.decode(v) for k, v in obj.items()}
        elif isinstance(obj, Sequence):
            return [self.decode(v) for v in obj]
        else:
            for typ, decoder in self._decoders.items():
                if isinstance(obj, typ):
                    return decoder(obj)
        return obj


AttrsEncoder = _AttrsEncoder()
AttrsEncoder.register_encoder(datetime, lambda dt: dt.isoformat())
AttrsEncoder.register_decoder(datetime, lambda s: datetime.fromisoformat(s))
AttrsEncoder.register_encoder(set, lambda s: list(s))
AttrsEncoder.register_decoder(set, lambda x: set(x))
AttrsEncoder.register_decoder(np.generic, lambda x: x.item())


class AttrsDict(H5Object[_MT], Mapping):
    """A dictionary-like object for the attributes of a group in the HDF5
    file.

    This object is tied to a simulation. If the simulation is read-only, the
    object is immutable. If mutable, changes are pushed to the HDF5 file
    immediately.

    Args:
        simulation: the simulation object
        path: path to the group in the HDF5 file
    """

    mutable: bool = False
    _path: str
    _dict: MutableMapping

    def __new__(cls, *args, **kwargs):
        if cls is not AttrsDict:
            return super().__new__(cls)

        # Singleton pattern for base attrs dict class
        # signature: __new__(cls, file: HDF5File[_MT], path: str)
        file = args[0] if args else kwargs.get("file")
        path = HDF5Path(args[1]) if len(args) > 1 else kwargs.get("path")

        instances = file._attrs_dict_instances
        if path not in instances:
            instances[path] = super().__new__(cls)
        return instances[path]

    def __init__(self, file: HDF5File[_MT], path: str):
        super().__init__(file)
        self._path = path
        self._dict = self.read()

    @with_file_open(FileMode.READ)
    def read(self) -> dict:
        return dict(AttrsEncoder.decode(self._obj.attrs))

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def _ipython_key_completions_(self):
        return tuple(self._dict.keys())

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self) -> str:
        return self._dict.__repr__()

    def __str__(self) -> str:
        return f"<AttrsDict(path={self._path})>"

    def _repr_pretty_(self, p, cycle):
        cls_name = type(self).__name__
        if cycle:
            p.text(f"{cls_name}(...)")
        else:
            with p.group(8, f"{cls_name}(", ")"):
                p.pretty(self._dict)

    @property
    def _obj(self) -> h5py.HLObject:
        obj = self._file[self._path]
        return obj

    @mutable_only
    def __setitem__(self: AttrsDict[Mutable], key: str, value: Any) -> None:
        self._dict[key] = value
        self.post_write_instruction(
            lambda: self._obj.attrs.__setitem__(key, AttrsEncoder.encode(value))
        )

    set = __setitem__
    """Set an attribute. This method is a an alias for `__setitem__`.

    Args:
        key: attribute name
        value: attribute value
    """

    @mutable_only
    def __delitem__(self: AttrsDict[Mutable], key: str) -> None:
        del self._dict[key]
        self.post_write_instruction(lambda: self._obj.attrs.__delitem__(key))

    @mutable_only
    def update(self: AttrsDict[Mutable], update_dict: dict) -> None:
        """Update the dictionary. This method pushes the update to the HDF5
        file.

        Args:
            update_dict: new dictionary
        """
        self._dict.update(update_dict)
        self.post_write_instruction(
            lambda: self._obj.attrs.update(AttrsEncoder.encode(update_dict))
        )
