from pathlib import Path
from typer.testing import CliRunner

from bamboost.cli.app import app
from bamboost.core.collection import Collection


def test_cli_run_create_dataclass(tmp_path: Path):
    # Set up paths
    collection_path = tmp_path / "my_collection"
    script_path = tmp_path / "my_script.py"

    # Write a test simulation script with a decorated dataclass
    script_content = """
from dataclasses import dataclass
from bamboost.cli.run import Script

script = Script()

@script.parameters
@dataclass
class Params:
    density: int = 123
    viscosity: float = 0.456
"""
    script_path.write_text(script_content)

    # Run the `create` subcommand
    runner = CliRunner()
    result = runner.invoke(app, ["run", "create", str(script_path), "--collection", str(collection_path)])

    output = result.stdout + result.stderr
    assert result.exit_code == 0, f"CLI command failed: {output}"
    assert "Created simulation" in output

    # Verify that the simulation was created and parameters were recorded
    coll = Collection(collection_path)
    assert len(coll) == 1
    sims = list(coll)
    assert len(sims) == 1
    sim = sims[0]
    assert sim.parameters["density"] == 123
    assert sim.parameters["viscosity"] == 0.456


def test_cli_run_create_function(tmp_path: Path):
    # Set up paths
    collection_path = tmp_path / "my_collection_func"
    script_path = tmp_path / "my_script_func.py"

    # Write a test simulation script with a decorated function returning a dict
    script_content = """
from bamboost.cli.run import Script

script = Script()

@script.parameters
def get_params():
    return {"density": 789, "viscosity": 0.111}
"""
    script_path.write_text(script_content)

    # Run the `create` subcommand
    runner = CliRunner()
    result = runner.invoke(app, ["run", "create", str(script_path), "--collection", str(collection_path)])

    output = result.stdout + result.stderr
    assert result.exit_code == 0, f"CLI command failed: {output}"
    assert "Created simulation" in output

    # Verify that the simulation was created and parameters were recorded
    coll = Collection(collection_path)
    assert len(coll) == 1
    sims = list(coll)
    assert len(sims) == 1
    sim = sims[0]
    assert sim.parameters["density"] == 789
    assert sim.parameters["viscosity"] == 0.111


def test_cli_run_create_and_local(tmp_path: Path):
    # Set up paths
    collection_path = tmp_path / "my_collection_local"
    script_path = tmp_path / "my_script_local.py"

    # Write a test simulation script with both @parameters, @main, and a custom stage
    script_content = """
from dataclasses import dataclass
from bamboost.cli.run import Script

script = Script()

@script.parameters
@dataclass
class Params:
    density: int = 456

@script.main
def run_sim(sim):
    sim.metadata["run_status"] = "run_completed"

@script.stage("post")
def post_sim(sim):
    sim.metadata["post_status"] = "post_completed"
"""
    script_path.write_text(script_content)

    runner = CliRunner()
    # 1. Create the simulation
    result_create = runner.invoke(
        app, ["run", "create", str(script_path), "--collection", str(collection_path)]
    )
    output_create = result_create.stdout + result_create.stderr
    assert result_create.exit_code == 0, f"Create failed: {output_create}"
    assert "Created simulation" in output_create

    # Get collection and simulation details
    coll = Collection(collection_path)
    assert len(coll) == 1
    sims = list(coll)
    sim = sims[0]
    assert sim.parameters["density"] == 456

    # 2. Run the simulation locally (default "main" stage)
    result_run = runner.invoke(
        app, ["run", "local-sim", "--uid", f"{coll.uid}:{sim.name}"]
    )
    output_run = result_run.stdout + result_run.stderr
    assert result_run.exit_code == 0, f"Run failed: {output_run}"
    assert "Executing stage 'main'" in output_run
    assert "Stage 'main' execution completed" in output_run

    # Verify that the simulation execution modified the metadata
    sim_reloaded = coll[sim.name]
    assert sim_reloaded.metadata["run_status"] == "run_completed"

    # 3. Run the custom "post" stage locally
    result_post = runner.invoke(
        app, ["run", "local-sim", "--uid", f"{coll.uid}:{sim.name}", "--stage", "post"]
    )
    output_post = result_post.stdout + result_post.stderr
    assert result_post.exit_code == 0, f"Post failed: {output_post}"
    assert "Executing stage 'post'" in output_post
    assert "Stage 'post' execution completed" in output_post

    # Verify that the post stage modified the metadata
    sim_final = coll[sim.name]
    assert sim_final.metadata["post_status"] == "post_completed"
