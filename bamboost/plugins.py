from __future__ import annotations

from abc import ABC
from itertools import chain
from typing import Any, ClassVar, Generic, Protocol, Type, TypedDict, TypeVar


class PluginProtocol(Protocol):
    name: str
    overwrite_classes: dict[str, type]


class BasePluginOpts(TypedDict):
    pass


_T_PluginOpts = TypeVar("_T_PluginOpts", bound=BasePluginOpts)


class PluginComponent(ABC):
    __source_plugin__: ClassVar[Plugin]
    __plugin_owned__: ClassVar[bool] = True


class Plugin(ABC, Generic[_T_PluginOpts]):
    name: ClassVar[str]
    overwrite_classes: dict[str, type]
    components: list[Type[PluginComponent]]

    def __init__(self, opts: _T_PluginOpts) -> None:
        self.opts = opts

        # remember the source plugin for each class
        for cls in chain(self.overwrite_classes.values(), self.components):
            setattr(cls, "__source_plugin__", self)


# active plugins set
_active: set[Plugin] = set()


def load(*plugins: Plugin) -> None:
    _active.update(plugins)


def unload() -> None:
    _active.clear()
