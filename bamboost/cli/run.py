from __future__ import annotations

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, overload

import typer
from typing_extensions import Annotated

from bamboost.cli.common import console

if TYPE_CHECKING:
    from bamboost.core.simulation import SimulationWriter

_workflow_registry: dict[str, Callable | None] = {
    "configure": None,
    "configure_plus": None,
    "execute": None,
}


DEFAULT_CREATE_SCRIPT = False
DEFAULT_SUBMIT = False


@overload
def configure(
    func: Callable[[], SimulationWriter], /
) -> Callable[[], SimulationWriter]: ...
@overload
def configure(
    *,
    create_default_run_script: bool = DEFAULT_CREATE_SCRIPT,
    submit: bool = DEFAULT_SUBMIT,
) -> Callable[[Callable[[], SimulationWriter]], Callable[[], SimulationWriter]]: ...
def configure(
    create_default_run_script: bool
    | Callable[[], SimulationWriter] = DEFAULT_CREATE_SCRIPT,
    submit: bool = DEFAULT_SUBMIT,
) -> Any:
    """Decorator to mark the configuration/setup function of a simulation."""

    def decorator(
        func: Callable[[], SimulationWriter],
    ) -> Callable[[], SimulationWriter]:
        _workflow_registry["configure"] = func
        func._bamboost_config_meta = {  # ty:ignore[unresolved-attribute]
            "create_default_run_script": (
                DEFAULT_CREATE_SCRIPT
                if callable(create_default_run_script)
                else create_default_run_script
            ),
            "submit": submit,
        }
        return func

    if callable(create_default_run_script):
        func = create_default_run_script
        return decorator(func)

    return decorator


def execute(func: Callable[[SimulationWriter], None]):
    """Decorator to mark the execution script of a simulation."""
    _workflow_registry["execute"] = func
    return func


class CommandChoice(str, Enum):
    CONFIGURE = "configure"
    EXECUTE = "execute"


def run(
    script_path: Annotated[
        Path, typer.Argument(..., help="Path to the simulation script.")
    ],
    command: Annotated[
        CommandChoice,
        typer.Argument(..., help="Subcommand to execute."),
    ],
    uid: Annotated[
        Optional[str],
        typer.Argument(help="Simulation UID to run (only for 'execute' subcommand)."),
    ] = None,
) -> None:
    """Run a decorated simulation script (either 'configure' or 'execute')."""
    script_path_abs = Path(script_path).resolve()
    if not script_path_abs.exists():
        console.print(f"[red]:cross_mark: Script file not found: {script_path_abs}")
        raise typer.Exit(1)

    # Dynamically import the user's script so the decorators register
    spec = importlib.util.spec_from_file_location("user_script", script_path_abs)
    if spec is None or spec.loader is None:
        console.print(f"[red]:cross_mark: Could not load script: {script_path_abs}")
        raise typer.Exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_script"] = module
    spec.loader.exec_module(module)

    if command == CommandChoice.CONFIGURE:
        fn = _workflow_registry.get("configure")
        if fn is None:
            console.print(
                "[red]:cross_mark: No configuration function decorated with @configure in the script."
            )
            raise typer.Exit(1)

        console.print("[blue]Running configure...")
        sim = fn()

        from bamboost.core.simulation import SimulationWriter

        if not isinstance(sim, SimulationWriter):
            console.print(
                "[red]:cross_mark: The configure function must return a SimulationWriter."
            )
            raise typer.Exit(1)

        meta = getattr(
            fn,
            "_bamboost_config_meta",
            {},
        )

        if meta.get("create_default_run_script", DEFAULT_CREATE_SCRIPT):
            sim.create_run_script(
                commands=[f"bamboost run {script_path_abs} execute {sim.uid}"]
            )
            console.print(
                f"[green]:heavy_check_mark: Created run script for simulation {sim.uid}"
            )

        if meta.get("submit", DEFAULT_SUBMIT):
            console.print("[blue]Submitting simulation to job scheduler...")
            sim.submit_simulation()
            console.print("[green]✅ Simulation submitted successfully.")

    elif command == CommandChoice.EXECUTE:
        fn = _workflow_registry.get("execute")
        if fn is None:
            console.print(
                "[red]:cross_mark: No execution function decorated with @execute in the script."
            )
            raise typer.Exit(1)

        if not uid:
            console.print(
                "[red]:cross_mark: Missing simulation UID for the 'execute' command."
            )
            raise typer.Exit(1)

        from bamboost.core.simulation import SimulationWriter

        sim = SimulationWriter.from_uid(uid)
        fn(sim)

    else:
        console.print(
            f"[red]:cross_mark: Unknown subcommand '{command}'. Choose 'configure' or 'execute'."
        )
        raise typer.Exit(1)
