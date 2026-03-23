from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import overload

from typing_extensions import Self

from bamboost import constants
from bamboost.mpi import Communicator


class CollectionUID(str):
    """UID of a collection. If no UID is provided, a new one is generated.

    Note:
        The generated UID is guaranteed to be unique across MPI ranks by broadcasting
        the generated UUID from the root rank.
    """

    def __new__(cls, uid: str | None = None, length: int = 10):
        uid = uid or cls.generate_uid(length)
        return super().__new__(cls, uid.upper())

    @staticmethod
    def generate_uid(length: int) -> str:
        if Communicator._active_comm.rank == 0:
            uid = uuid.uuid4().hex[:length].upper()
        else:
            uid = ""
        uid: str = Communicator._active_comm.bcast(uid, root=0)
        return uid


class SimulationName(str):
    """Name of a simulation. If no name is provided, a new one is generated.

    Args:
        name (Optional[str]): The desired name for the simulation. If not provided,
            a unique name will be generated.
        length (int): The length of the generated name if `name` is not provided.
            Default is 10.

    Note:
        The generated name is guaranteed to be unique across MPI ranks by broadcasting
        the generated UUID from the root rank.
    """

    def __new__(cls, name: str | None = None, length: int = 10):
        name = name or cls.generate_name(length)
        return super().__new__(cls, name)

    @staticmethod
    def generate_name(length: int) -> str:
        if Communicator._active_comm.rank == 0:
            uid = uuid.uuid4().hex[:length]
        else:
            uid = ""
        uid: str = Communicator._active_comm.bcast(uid, root=0)
        return uid


@dataclass(frozen=True, init=False)  # init=False because we handle it in __new__
class SimulationUID:
    """UID of a simulation, consisting of the collection UID and the simulation name.

    Use `str(SimulationUID(...))` to get the string representation of the UID, which is in
    the format `<collection_uid>:<simulation_name>`. The constructor can be called with
    either the string representation or the collection UID and simulation name as separate
    arguments.
    """

    collection_uid: CollectionUID
    simulation_name: SimulationName

    @overload
    def __new__(cls, uid: str | SimulationUID, /) -> Self: ...

    @overload
    def __new__(
        cls,
        collection_uid: CollectionUID | str,
        simulation_name: SimulationName | str,
        /,
    ) -> Self: ...

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], SimulationUID):
            return args[0]

        instance = super().__new__(cls)

        if len(args) == 1 and isinstance(args[0], str):
            try:
                c_uid, s_name = args[0].split(constants.UID_SEPARATOR, 1)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid SimulationUID string: {args[0]!r}. expected format "
                    f"'<collection_uid>{constants.UID_SEPARATOR}<simulation_name>'"
                ) from exc
            object.__setattr__(instance, "collection_uid", CollectionUID(c_uid))
            object.__setattr__(instance, "simulation_name", SimulationName(s_name))

        elif len(args) == 2:
            object.__setattr__(instance, "collection_uid", CollectionUID(args[0]))
            object.__setattr__(instance, "simulation_name", SimulationName(args[1]))

        else:
            raise ValueError("Invalid arguments for SimulationUID")

        return instance

    def __eq__(self, other):
        if isinstance(other, SimulationUID):
            return (
                self.collection_uid == other.collection_uid
                and self.simulation_name == other.simulation_name
            )
        if isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __str__(self):
        return f"{self.collection_uid}{constants.UID_SEPARATOR}{self.simulation_name}"
