from __future__ import annotations

from enum import Enum
from pathlib import Path
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import typer
from typing_extensions import Annotated

from bamboost import config
from bamboost.cli import _completion
from bamboost.cli.common import console
from bamboost.cli.run.script import Script as Script
from bamboost.cli.run.script import _get_script

if TYPE_CHECKING:
    from bamboost.core.simulation import SimulationWriter

T_Callable = TypeVar("T_Callable", bound=Callable[..., Any])
T_Param = TypeVar("T_Param", bound=Callable[..., Any] | type[Any])


app_run = typer.Typer(
    name="run",
    help="Run a simulation script with decorated functions for configuration and execution.",
    no_args_is_help=True,
)


@app_run.callback()
def _run(
    mpi: Annotated[
        bool,
        typer.Option(
            "--mpi",
            help="Whether to run the simulation with MPI support. This will enable MPI support in bamboost",
        ),
    ] = False,
) -> None:
    """Main entry point for the run command group."""
    if mpi:
        config.options.mpi = True


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
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            "-s",
            help="Whether to submit the simulation to the cluster after creation using the slurm config.",
        ),
    ] = False,
) -> None:
    """Create a simulation in a collection using script parameters."""
    script = _get_script(entry_point)
    sim = _create(script, entry_point, collection_path, duplicate_action)

    if submit:
        for s in sim:
            job_id = _submit(s, script)


def _create(
    script: Script,
    entry_point: Path,
    collection_path: Path,
    duplicate_action: DuplicateAction | None = None,
) -> tuple[SimulationWriter, ...]:
    """Internal function to create a simulation from a script."""

    parameters_fn = script._stages.parameters
    if parameters_fn is None:
        console.print(
            "[red]:cross_mark: No parameters function or dataclass registered on the Script object."
        )
        raise typer.Exit(1)

    try:
        params = parameters_fn()
        if not isinstance(params, tuple):
            # If it's a single dict, wrap it in a tuple to make it consistent for downstream processing
            params = (params,)
    except Exception as e:
        console.print(f"[red]:cross_mark: Failed to instantiate/call parameters: {e}")
        raise typer.Exit(1)

    from bamboost.core import Collection

    _duplicate_action = (
        duplicate_action.value if duplicate_action else script._duplicate_action
    )
    coll = Collection(collection_path, create_if_not_exist=True)
    sims = []

    for p in params:
        if not isinstance(p, dict):
            console.print(
                f"[red]:cross_mark: Parameters function must return a dict or a tuple of dicts, got {type(p).__name__}."
            )
            raise typer.Exit(1)
        try:
            sim = coll.add(
                parameters=p,
                duplicate_action=_duplicate_action,
                entry_point=entry_point,
            )
            sims.append(sim)
        except Exception as e:
            console.print(f"[red]:cross_mark: Failed to create simulation: {e}")

    if not sims:
        console.print(
            "[red]:cross_mark: No simulations were created. Please check the parameters function and duplicate action."
        )
        raise typer.Exit(1)

    console.print(
        f"[green]:heavy_check_mark: Created simulation(s) {', '.join(str(s.uid) for s in sims)} in collection '{coll.uid}'."
    )

    return tuple(sims)


@app_run.command()
def main(
    uid: Annotated[
        str,
        typer.Option(
            "--uid",
            help="UID or alias of the simulation to run.",
            autocompletion=_completion._get_collection_sim_completion,
        ),
    ],
) -> None:
    """Run the main stage of a simulation by UID."""
    from bamboost.core.simulation import SimulationWriter

    sim = SimulationWriter.from_uid(uid)
    script = _get_script(sim.files["entry_point.py"])

    console.print("[blue]Executing stage 'main'...")
    stage_fn = script._stages.main
    if stage_fn is None:
        console.print(
            "[red]:cross_mark: No stage function 'main' registered on the Script object."
        )
        raise typer.Exit(1)

    # context manager to set status="running" and "finished" after execution
    with sim:
        stage_fn(sim)

    console.print("[green]✅ Stage 'main' execution completed.")


@app_run.command()
def start(
    uid: Annotated[
        str,
        typer.Option(
            "--uid",
            help="UID or alias of the simulation to run.",
            autocompletion=_completion._get_collection_sim_completion,
        ),
    ],
) -> None:
    """Spawn a new process to run the local config stage of a simulation by UID."""
    from bamboost.core.simulation import SimulationWriter

    sim = SimulationWriter.from_uid(uid)
    script = _get_script(sim.files["entry_point.py"])

    console.print("[blue]Spawning process to execute stage 'local'...")

    get_local_config = script._run_config.local
    if get_local_config is None:
        console.print(
            "No local run configuration found on the Script object. Using default execution of 'main' stage instead."
        )
        bash_instructions = f"bamboost run main --uid {uid}"
    else:
        bash_instructions = get_local_config(sim.uid)

    import os
    import subprocess

    subprocess_command = f"bash -c '{bash_instructions}'"
    subprocess.run(subprocess_command, shell=True, check=True, env=os.environ)


@app_run.command()
def submit(
    uid: Annotated[
        str,
        typer.Option(
            "--uid",
            help="UID or alias of the simulation to submit.",
            autocompletion=_completion._get_collection_sim_completion,
        ),
    ],
) -> None:
    """Submit a simulation by UID to the cluster using the slurm config."""
    from bamboost.core.simulation import SimulationWriter

    sim = SimulationWriter.from_uid(uid)
    script = _get_script(sim.files["entry_point.py"])
    _submit(sim, script)


def _submit(sim: SimulationWriter, script: Script) -> str:
    get_slurm_config = script._run_config.slurm
    if get_slurm_config is None:
        console.print(
            "[red]:cross_mark: No slurm run configuration found on the Script object."
        )
        raise typer.Exit(1)

    slurm_instructions = get_slurm_config(sim.uid)
    subprocess_command = f"sbatch --parsable --wrap='{slurm_instructions}'"

    import os
    import subprocess

    try:
        with console.status("[bold green]Submitting simulation to cluster..."):
            res = subprocess.run(
                subprocess_command,
                shell=True,
                check=True,
                env=os.environ,
                capture_output=True,
                text=True,
            )
            # sbatch returns the job ID on stdout when --parsable is used
            job_id = res.stdout.strip()
            console.print(f"[green]✅ Simulation submitted with Job ID: {job_id}")
            return job_id

    except CalledProcessError as e:
        console.print("\n[bold red]❌ Cluster Submission Failed![/bold red]")
        # e.stderr contains the exact error message from sbatch
        console.print(f"[dim]{e.stderr.strip()}[/dim]")
        raise typer.Exit(code=1)  # Cleanly exit Typer


@app_run.command()
def local_sim(
    uid: Annotated[
        str,
        typer.Option(
            "--uid",
            help="UID or alias of the simulation to submit.",
            autocompletion=_completion._get_collection_sim_completion,
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

    sim = SimulationWriter.from_uid(uid)
    script = _get_script(sim.files["entry_point.py"])

    console.print(f"[blue]Executing stage '{stage}'...")
    stage_fn = script._stages.get(stage)
    if stage_fn is None:
        console.print(
            f"[red]:cross_mark: No stage function '{stage}' registered on the Script object."
        )
        raise typer.Exit(1)

    # context manager to set status="running" and "finished" after execution
    with sim:
        stage_fn(sim)

    console.print(f"[green]✅ Stage '{stage}' execution completed.")


@app_run.command()
def local(
    entry_point: Annotated[
        Path, typer.Argument(..., help="Path to the simulation script.")
    ],
    collection_uid: Annotated[
        str | None,
        typer.Option(
            "--collection",
            "-c",
            help="UID or alias of the collection to run the simulation from if it already exists.",
            autocompletion=_completion._get_uids_from_db,
        ),
    ] = None,
    collection_path: Annotated[
        Path | None,
        typer.Option(
            "--collection-path",
            "-cp",
            help="Path to the collection where the simulation should be created if not already existing.",
            show_default=False,
        ),
    ] = None,
    duplicate_action: Annotated[
        DuplicateAction | None,
        typer.Option(
            "--duplicate",
            "-d",
            help="Action to take if a simulation with the same parameters already exists when creating a new simulation.",
        ),
    ] = None,
) -> None:
    """Run a simulation script locally."""
    script = _get_script(entry_point)
    from bamboost.core.simulation import SimulationWriter
    from bamboost.index.scanner import find_uid_from_path

    sim = None
    if collection_uid is None and collection_path is None:
        script_dir = entry_point.parent
        possible_coll_uid = find_uid_from_path(script_dir.parent)
        if possible_coll_uid is not None:
            sim = SimulationWriter.from_uid(f"{possible_coll_uid}:{script_dir.name}")
            console.print(
                f"[blue]Running simulation from collection '{possible_coll_uid}' with name '{script_dir.name}'..."
            )

    if sim is None:
        from bamboost.index import Index

        if collection_uid:
            collection_path = Index.default.resolve_path(collection_uid)
        elif collection_path:
            collection_path = collection_path.resolve()
        elif script._collection_uid:
            collection_path = Index.default.resolve_path(script._collection_uid)

        if collection_path is None:
            console.print(
                "[red]:cross_mark: No collection UID found in the script path and no collection path provided."
            )
            raise typer.Exit(1)

        sims = _create(script, entry_point, collection_path, duplicate_action)
        # for now, we only run the first simulation if multiple were created from the parameters function
        sim = sims[0]

    console.print("[blue]Executing stage 'main'...")
    stage_fn = script._stages.main
    if stage_fn is None:
        console.print(
            "[red]:cross_mark: No stage function 'main' registered on the Script object."
        )
        raise typer.Exit(1)

    # context manager to set status="running" and "finished" after execution
    with sim:
        stage_fn(sim)

    console.print("[green]✅ Stage 'main' execution completed.")
