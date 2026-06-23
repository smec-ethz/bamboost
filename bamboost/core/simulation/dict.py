from __future__ import annotations

from typing import TYPE_CHECKING, MutableMapping

from bamboost import constants
from bamboost.core.hdf5.attrsdict import AttrsDict, mutable_only
from bamboost.core.hdf5.file import _MT, Mutable
from bamboost.core.utilities import SimulationUID
from bamboost.exceptions import ForbiddenParameterKeyError

if TYPE_CHECKING:
    from bamboost.core.simulation.base import _Simulation


RESERVED_KEYS = {
    "id",
    "collection_uid",
    "name",
    "created_at",
    "modified_at",
    "description",
    "tags",
    "status",
    "submitted",
    "links",
}


def validate_parameter_key(key: str) -> None:
    """Validate a parameter key.

    Args:
        key: The key to validate.

    Raises:
        ValueError: If the key is reserved or contains a period.
    """
    # only top-level keys are checked for reservation and periods.
    # nested keys (e.g., 'params.nested_key') are already handled by the flattening logic
    # but we want to prevent users from manually passing a key with a period
    # which would interfere with our flattening/unflattening.
    if key in RESERVED_KEYS:
        raise ForbiddenParameterKeyError(f"Parameter key '{key}' is reserved.")
    if "." in key:
        raise ForbiddenParameterKeyError(
            f"Parameter key '{key}' cannot contain a period ('.')."
        )


# TODO: need to update for new backend
class Links(AttrsDict[_MT]):
    _simulation: _Simulation
    _dict: MutableMapping[str, SimulationUID]

    def __init__(self, simulation: _Simulation[_MT]) -> None:
        super().__init__(simulation._file, constants.PATH_LINKS)
        self._simulation = simulation

    def read(self) -> dict[str, SimulationUID]:
        return {key: SimulationUID(value) for key, value in super().read().items()}

    def __getitem__(self, key: str) -> "_Simulation":
        from bamboost.core.simulation import Simulation

        return Simulation.from_uid(super().__getitem__(key))

    @mutable_only
    def __setitem__(self: Links[Mutable], key: str, value: str | SimulationUID) -> None:
        self.update({key: value})

    @mutable_only
    def update(
        self: Links[Mutable], update_dict: MutableMapping[str, SimulationUID | str]
    ) -> None:
        # update values to ensure they are all SimulationUIDs
        _update_dict = {key: SimulationUID(value) for key, value in update_dict.items()}

        # check sql first to avoid writing to hdf5 file if the update is not valid
        self._simulation.update_database(links=_update_dict)

        self._dict.update(_update_dict)
        self.post_write_instruction(
            # in the hdf5 file, we store the links as strings
            lambda: self._obj.attrs.update(
                {key: str(value) for key, value in _update_dict.items()}
            )
        )
