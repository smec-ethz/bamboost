from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

import typer
from typing_extensions import Annotated

from bamboost.cli import _completion
from bamboost.cli.common import console
from bamboost.cli.run.script import Script as Script
from bamboost.cli.run.script import _get_script

T_Callable = TypeVar("T_Callable", bound=Callable[..., Any])
T_Param = TypeVar("T_Param", bound=Callable[..., Any] | type[Any])


app_run = typer.Typer(
    name="run",
    help="Run a simulation script with decorated functions for configuration and execution.",
    no_args_is_help=True,
)


class DuplicateAction(str, Enum):
    IGNORE = "ignore"
    REPLACE = "replace"
    SKIP = "skip"
    RAISE = "raise"


@app_run.command()
def create(
    entry_point: Annotated[
        Path, typer.Argument(..., help="Path to the simulation script.")
    ],
    collection_path: Annotated[
        Path,
        typer.Option(
            "--collection",
            "-c",
            help="Path to the collection where the simulation should be created.",
            show_default=False,
        ),
    ],
    duplicate_action: Annotated[
        DuplicateAction | None,
        typer.Option(
            "--duplicate",
            "-d",
            help="Action to take if a simulation with the same parameters already exists.",
        ),
    ] = None,
) -> None:
    """Create a simulation in a collection using script parameters."""
    script = _get_script(entry_point)

    try:
        parameters_fn = script._stages.get("parameters")
        if parameters_fn is None:
            console.print(
                "[red]:cross_mark: No parameters function or dataclass registered on the Script object."
            )
            raise typer.Exit(1)

        try:
            params = parameters_fn()
        except Exception as e:
            console.print(
                f"[red]:cross_mark: Failed to instantiate/call parameters: {e}"
            )
            raise typer.Exit(1)

        from bamboost.core import Collection

        _duplicate_action = (
            duplicate_action.value if duplicate_action else script._duplicate_action
        )

        coll = Collection(collection_path, create_if_not_exist=True)
        sim = coll.add(
            parameters=params,
            duplicate_action=_duplicate_action,
            entry_point=entry_point,
        )
    except Exception as e:
        console.print(f"[red]:cross_mark: Failed to create simulation: {e}")
        raise typer.Exit(1)

    console.print(f"[green]:heavy_check_mark: Created simulation {sim.uid}")


@app_run.command()
def local(
    collection_uid: Annotated[
        str,
        typer.Argument(
            ...,
            help="UID or alias of the collection to run the simulation from.",
            autocompletion=_completion._get_uids_from_db,
        ),
    ],
    simulation_name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Name of the simulation to run.",
            autocompletion=_completion._get_simulation_names,
        ),
    ],
    stage: Annotated[
        str,
        typer.Option(
            "--stage",
            "-s",
            help="Name of the stage to execute.",
        ),
    ] = "main",
) -> None:
    """Run the simulation locally."""
    from bamboost.core.simulation import SimulationWriter

    sim = SimulationWriter.from_uid(f"{collection_uid}:{simulation_name}")
    script = _get_script(sim.files["entry_point.py"])

    console.print(f"[blue]Executing stage '{stage}'...")
    stage_fn = script._stages.get(stage)
    if stage_fn is None:
        console.print(
            f"[red]:cross_mark: No stage function '{stage}' registered on the Script object."
        )
        raise typer.Exit(1)

    stage_fn(sim)
    console.print(f"[green]✅ Stage '{stage}' execution completed.")
