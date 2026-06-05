from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, overload

import typer

from bamboost.cli.common import console

T_Callable = TypeVar("T_Callable", bound=Callable[..., Any])
T_Dataclass = TypeVar("T_Dataclass", bound=type[Any])
T_Param = TypeVar("T_Param", bound=Callable[..., Any | tuple[Any, ...]])

if TYPE_CHECKING:
    from bamboost.core.simulation import SimulationWriter
    from bamboost.index import SimulationUID

    _FnMain = Callable[[SimulationWriter], Any]
    _FnConfig = Callable[[SimulationUID], str]


@dataclass
class _Stages:
    parameters: Callable[..., dict | tuple[dict]] | None = None
    main: _FnMain | None = None
    custom: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def get(self, name: str) -> Callable[..., Any] | None:
        if name == "parameters":
            return self.parameters
        elif name == "main":
            return self.main
        else:
            return self.custom.get(name)


@dataclass
class _RunConfig:
    slurm: _FnConfig | None = None
    local: _FnConfig | None = None


class Script:
    """Registry object for registering simulation parameters and execution stages."""

    def __init__(
        self,
        *,
        duplicate_action: Literal["raise", "replace", "skip", "ignore"] = "raise",
        collection_uid: str | None = None,
    ) -> None:
        self._duplicate_action = duplicate_action
        self._collection_uid: str | None = collection_uid

        self._stages: _Stages = _Stages()
        self._run_config: _RunConfig = _RunConfig()

    def stage(self, name: str) -> Callable[[T_Callable], T_Callable]:
        """Decorator to register a custom stage function by name."""

        def decorator(fn: T_Callable) -> T_Callable:
            self._stages.custom[name] = fn
            return fn

        return decorator

    @overload
    def parameters(self, fn: T_Param, /) -> T_Param: ...
    @overload
    def parameters(self, cls: T_Dataclass, /) -> T_Dataclass: ...
    def parameters(self, fn_or_cls: T_Dataclass | T_Param, /) -> T_Dataclass | T_Param:
        """Decorator to register the parameters function or dataclass."""
        if not (callable(fn_or_cls) or is_dataclass(fn_or_cls)):
            console.print(
                f"[red]:cross_mark: @parameters can only be applied to a function or dataclass, got {type(fn_or_cls).__name__}."
            )
            raise typer.Exit(1)

        def _asdict(*args, **kwargs) -> dict:
            params = fn_or_cls(*args, **kwargs)

            if isinstance(params, tuple):
                return tuple(asdict(p) if is_dataclass(p) else p for p in params)  # type: ignore

            if is_dataclass(params):
                return asdict(params)
            elif isinstance(params, dict):
                return params
            else:
                console.print(
                    f"[red]:cross_mark: Parameters must be a dictionary or a dataclass, got {type(params).__name__}."
                )
                raise typer.Exit(1)

        self._stages.parameters = _asdict
        return fn_or_cls

    def main(self, fn: _FnMain) -> _FnMain:
        """Decorator to register the main execution stage function."""
        self._stages.main = fn
        return fn

    # Config script decorators for different execution environments
    def config_slurm(self, fn: _FnConfig) -> _FnConfig:
        """Decorator to register a function that returns Slurm-specific configuration."""
        self._run_config.slurm = fn
        return fn

    def config_local(self, fn: _FnConfig) -> _FnConfig:
        """Decorator to register a function that returns local execution configuration."""
        self._run_config.local = fn
        return fn


def _import_script(script_path: Path) -> Any:
    script_path_abs = Path(script_path).resolve()
    if not script_path_abs.exists():
        console.print(f"[red]:cross_mark: Script file not found: {script_path_abs}")
        raise typer.Exit(1)

    # Dynamically import the user's script
    spec = importlib.util.spec_from_file_location("user_script", script_path_abs)
    if spec is None or spec.loader is None:
        console.print(f"[red]:cross_mark: Could not load script: {script_path_abs}")
        raise typer.Exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_script"] = module
    spec.loader.exec_module(module)
    return module


def _find_script(module: Any) -> Script:
    """Helper to search for an instance of Script in the imported module."""
    for name in dir(module):
        attr = getattr(module, name)
        if isinstance(attr, Script):
            return attr
    console.print(
        "[red]:cross_mark: No Script instance found in the simulation script."
    )
    raise typer.Exit(1)


def _get_script(script_path: Path) -> Script:
    module = _import_script(script_path)
    script = _find_script(module)
    return script
