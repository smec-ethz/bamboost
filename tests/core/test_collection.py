import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from bamboost.core.collection import Collection
from bamboost.core.simulation.base import Simulation, SimulationWriter
from bamboost.exceptions import DuplicateSimulationError
from bamboost.index import get_identifier_filename


def _populate_basic_collection(collection: Collection) -> None:
    collection.add(
        name="testsim1",
        parameters={"first_name": "John", "age": 20},
        override=True,
    )
    collection.add(
        name="testsim2",
        parameters={"first_name": "Jane", "age": 30},
        override=True,
    )
    collection.add(
        name="testsim3",
        parameters={"first_name": "Jack", "age": 40},
        override=True,
    )


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
    sim1 = tmp_collection_burn.add(name="override_test", parameters={"param": 1})

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
    _ = tmp_collection_burn.add(name="problematic_sim", parameters=parameters)
    with pytest.raises(DuplicateSimulationError):
        _ = tmp_collection_burn.add(parameters=parameters, duplicate_action="raise")

    assert len(tmp_collection_burn) == 1  # Only one simulation should exist


def test_collection_filter_combines_conditions(tmp_collection_burn: Collection):
    _populate_basic_collection(tmp_collection_burn)

    filtered = tmp_collection_burn.filter(
        (tmp_collection_burn.k["age"] == 20) | (tmp_collection_burn.k["age"] == 40)
    )
    items = list(filtered)
    assert {sim.name for sim in items} == {"testsim1", "testsim3"}
    assert all(isinstance(sim, Simulation) for sim in items)

    chained = filtered.filter(filtered.k["first_name"] == "Jack")
    chained_items = list(chained)
    assert [sim.name for sim in chained_items] == ["testsim3"]
    assert all(isinstance(sim, Simulation) for sim in chained_items)

    assert len(tmp_collection_burn) == 3


def test_collection_sort_orders_simulations(tmp_collection_burn: Collection):
    _populate_basic_collection(tmp_collection_burn)

    sorted_coll = tmp_collection_burn.sort(
        tmp_collection_burn.k["age"], ascending=False
    )
    items = list(sorted_coll)
    assert [sim.name for sim in items] == ["testsim3", "testsim2", "testsim1"]
    assert all(isinstance(sim, Simulation) for sim in items)
    assert len(tmp_collection_burn) == 3


def test_collection_filter_then_sort_iteration(tmp_collection_burn: Collection):
    _populate_basic_collection(tmp_collection_burn)

    chained = tmp_collection_burn.filter(tmp_collection_burn.k["age"] >= 30).sort(
        "first_name", ascending=False
    )
    items = list(chained)
    assert [sim.name for sim in items] == ["testsim2", "testsim3"]
    assert all(isinstance(sim, Simulation) for sim in items)
    # iteration should yield fresh Simulation objects each time
    new_items = list(chained)
    assert [sim.name for sim in new_items] == ["testsim2", "testsim3"]
    assert len(tmp_collection_burn) == 3


def test_collection_multi_sort(tmp_collection_burn: Collection):
    tmp_collection_burn.add(
        name="simA", parameters={"priority": 1, "step": 2}, override=True
    )
    tmp_collection_burn.add(
        name="simB", parameters={"priority": 1, "step": 1}, override=True
    )
    tmp_collection_burn.add(
        name="simC", parameters={"priority": 2, "step": 1}, override=True
    )

    sorted_coll = tmp_collection_burn.sort("priority").sort("step", ascending=False)
    items = list(sorted_coll)
    assert [sim.name for sim in items] == ["simA", "simB", "simC"]
    assert all(isinstance(sim, Simulation) for sim in items)


def test_collection_metadata_save_persists_yaml(tmp_collection_burn: Collection):
    meta = tmp_collection_burn.metadata
    meta.tags = ["alpha", "beta"]
    meta.aliases = ["Primary"]
    meta.description = "demo collection"
    meta.update({"custom": {"foo": "bar"}})

    meta.save()

    metadata_file = tmp_collection_burn.path / get_identifier_filename(
        tmp_collection_burn.uid
    )
    stored = yaml.safe_load(metadata_file.read_text())

    assert stored["uid"] == tmp_collection_burn.uid
    assert stored["tags"] == ["alpha", "beta"]
    assert stored["aliases"] == ["Primary"]
    assert stored["description"] == "demo collection"
    assert stored["custom"] == {"foo": "bar"}
    assert isinstance(stored["created_at"], datetime)


def test_collection_metadata_loads_existing_yaml(tmp_collection_burn: Collection):
    metadata_file = tmp_collection_burn.path / get_identifier_filename(
        tmp_collection_burn.uid
    )
    metadata_file.write_text(
        "\n".join(
            [
                f"uid: {tmp_collection_burn.uid}",
                "created_at: 2024-01-02T03:04:05",
                "tags:",
                "  - foo",
                "  - foo",
                "  - bar",
                "  - ''",
                "aliases:",
                "  - ALPHA",
                "  - alpha",
                "custom: 7",
            ]
        )
    )

    tmp_collection_burn.__dict__.pop("metadata", None)
    meta = tmp_collection_burn.metadata

    assert meta.tags == ["foo", "bar"]
    assert meta.aliases == ["ALPHA"]
    assert meta["custom"] == 7
    assert meta.created_at == datetime(2024, 1, 2, 3, 4, 5)
