from __future__ import annotations

from typing import ClassVar, Generic, Type, TypedDict, TypeVar


class ElligibleForPlugin:
    """A base class for all classes that should be available to be replaced by plugins."""

    def __new__(cls, *args, **kwargs):
        # hook up plugins here
        for plugin in _active:
            if cls in plugin.override_components:
                return super().__new__(plugin.override_components[cls])

        return super().__new__(cls)


class PluginComponent:
    __plugin__: ClassVar[Plugin]

    def __init_subclass__(cls, **kwargs) -> None:
        cls.__replace_base__ = kwargs.pop("replace_base", False)
        return super().__init_subclass__()


class PluginMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Collect and register plugin components
        _components = []
        _replacements: dict[type, type] = {}

        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, type) and issubclass(attr_value, PluginComponent):
                _components.append(attr_value)

                # Handle replacement if `replace_base=True` is set
                if getattr(attr_value, "__replace_base__", False):
                    for base in attr_value.__bases__:
                        if issubclass(base, ElligibleForPlugin):
                            _replacements[base] = attr_value
                            break
                    else:
                        raise ValueError(
                            f"Could not find a base class that is elligible for replacement by the plugin API (for {attr_value})"
                        )

        namespace["components"] = _components
        namespace["override_components"] = _replacements

        return super().__new__(mcs, name, bases, namespace)


class BasePluginOpts(TypedDict):
    pass


_T_PluginOpts = TypeVar("_T_PluginOpts", bound=BasePluginOpts)


class Plugin(Generic[_T_PluginOpts], metaclass=PluginMeta):
    name: str
    override_components: dict[type, type]
    components: list[Type[PluginComponent]]

    def __init__(self, opts: _T_PluginOpts) -> None:
        self.opts = opts

        # remember the source plugin for each class
        for cls in self.components:
            cls.__plugin__ = self


# active plugins set
_active: set[Plugin] = set()


def load(*plugins: Plugin) -> None:
    _active.update(plugins)


def unload() -> None:
    _active.clear()
