import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from bamboost.core.collection import Collection
from bamboost.core.simulation.base import Simulation, SimulationWriter
from bamboost.exceptions import DuplicateSimulationError

# ------------------- Fixtures -------------------
# also used:
# fixture 'tmp_collection' from conftest.py


# ------------------- Tests -------------------


def test_if_new_collection_created(tmp_collection: Collection):
    assert tmp_collection.path.exists()


def test_length(test_collection: Collection):
    assert len(test_collection) == 3


def test_getitem(test_collection: Collection):
    from bamboost.core.simulation import Simulation

    sim1 = test_collection["testsim1"]
    assert isinstance(sim1, Simulation)
    assert sim1.name == "testsim1"


def test_df(test_collection: Collection):
    df = test_collection.df
    # size matching?
    assert df.index.size == 3
    # parameters exist in dataframe?
    test_collection._index.upsert_simulation(test_collection.uid, "testsim1")
    assert {"first_name", "age", "list", "dict.a", "dict.b"}.issubset(set(df.columns))
    # names correct?
    assert set(df["name"].tolist()) == {"testsim1", "testsim2", "testsim3"}


@pytest.mark.parametrize(
    "name,parameters",
    [("testsim1", {"param1": 1, "param2": "value"}), (None, {"param1": 1})],
)
def test_create_simulation_basic(tmp_collection_burn: Collection, name, parameters):
    sim_writer = tmp_collection_burn.add(name=name, parameters=parameters)
    assert isinstance(sim_writer, SimulationWriter)
    assert sim_writer.parameters._dict == parameters

    # assert that folder exists
    assert sim_writer.path.exists()


def test_create_simulation_with_description(tmp_collection: Collection):
    description = "This is a test simulation."
    sim_writer = tmp_collection.add(
        name="described_sim",
        parameters={"param": 1},
        description=description,
    )
    assert sim_writer.metadata["description"] == description


def test_create_simulation_with_files(tmp_collection: Collection, tmp_path: Path):
    # Create a temp file to "copy" into simulation
    temp_file = tmp_path / "testfile.txt"
    temp_file.write_text("Test content")

    sim_writer = tmp_collection.add(
        name="sim_with_file",
        parameters={"param": 1},
        files=[str(temp_file)],
        override=True,
    )

    # Check if file was copied
    sim_folder = sim_writer.path
    assert (sim_folder / "testfile.txt").exists()
    assert (sim_folder / "testfile.txt").read_text() == "Test content"


def test_create_simulation_with_links(tmp_collection: Collection):
    other = tmp_collection.add()
    sim_writer = tmp_collection.add(
        name="sim_with_links",
        parameters={"param": 1},
        links={"linked_sim": other.uid},
        duplicate_action="ignore",
    )
    assert isinstance(sim_writer.links["linked_sim"], Simulation)
    assert sim_writer.links["linked_sim"].uid == other.uid


def test_create_simulation_overrides_existing(tmp_collection_burn: Collection):
    # Create initial simulation
    sim1 = tmp_collection_burn.add(
        name="override_test", parameters={"param": 1}
    )

    # Create it again with override=True
    sim2 = tmp_collection_burn.add(
        name="override_test", parameters={"param": 2}, override=True
    )

    sim_final = tmp_collection_burn["override_test"]

    assert len(tmp_collection_burn) == 1  # Only one simulation should exist
    assert sim_final.name == "override_test"  # Same name reused
    assert sim_final.parameters == {"param": 2}


@pytest.mark.mpi_skip(reason="Test patching does not work with MPI")
def test_create_simulation_error_handling(tmp_collection: Collection):
    with patch("shutil.rmtree", wraps=shutil.rmtree) as spy_rmtree:
        with patch(
            "bamboost.core.simulation.base.SimulationWriter.initialize",
            side_effect=PermissionError("Sim init failed"),
        ):
            with pytest.raises(PermissionError, match="Sim init failed"):
                tmp_collection.add(
                    name="error_sim", parameters={"param": 1}, override=True
                )

        # Ensure cleanup was attempted
        spy_rmtree.assert_called_once()
        assert not (tmp_collection.path / "error_sim").exists()


@pytest.mark.parametrize(
    "parameters",
    [
        {"int": 1, "string": "value", "float": 3.14},
        {"list": [1, 2, 3]},
        {"dict": {"a": 1, "b": 2}},
        {"nested": {"list": [1, 2], "dict": {"key": "value"}}},
    ],
)
def test_create_simulation_duplicate_raises_exception(
    tmp_collection_burn: Collection, parameters: dict
):
    _ = tmp_collection_burn.add(
        name="problematic_sim", parameters=parameters
    )
    with pytest.raises(DuplicateSimulationError):
        _ = tmp_collection_burn.add(
            parameters=parameters, duplicate_action="raise"
        )

    assert len(tmp_collection_burn) == 1  # Only one simulation should exist
