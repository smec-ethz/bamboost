from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import typer

from bamboost.cli.common import console

T_Callable = TypeVar("T_Callable", bound=Callable[..., Any])
T_Param = TypeVar("T_Param", bound=type[Any])


class Script:
    """Registry object for registering simulation parameters and execution stages."""

    def __init__(
        self,
        *,
        duplicate_action: Literal["raise", "replace", "skip", "ignore"] = "raise",
    ) -> None:
        self._duplicate_action = duplicate_action
        self._stages: dict[str, Callable[..., Any]] = {}

    def stage(self, name: str) -> Callable[[T_Callable], T_Callable]:
        """Decorator to register a custom stage function by name."""

        def decorator(fn: T_Callable) -> T_Callable:
            self._stages[name] = fn
            return fn

        return decorator

    def parameters(self, fn_or_cls: T_Param) -> T_Param:
        """Decorator to register the parameters function or dataclass."""
        if not (callable(fn_or_cls) or is_dataclass(fn_or_cls)):
            console.print(
                f"[red]:cross_mark: @parameters can only be applied to a function or dataclass, got {type(fn_or_cls).__name__}."
            )
            raise typer.Exit(1)

        def _asdict(*args, **kwargs) -> dict:
            params = fn_or_cls(*args, **kwargs)

            if is_dataclass(params):
                return asdict(params)
            elif isinstance(params, dict):
                return params
            else:
                console.print(
                    f"[red]:cross_mark: Parameters must be a dictionary or a dataclass, got {type(params).__name__}."
                )
                raise typer.Exit(1)

        self._stages["parameters"] = _asdict
        return fn_or_cls

    def main(self, fn: T_Callable) -> T_Callable:
        """Decorator to register the main execution stage function."""
        self._stages["main"] = fn
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
