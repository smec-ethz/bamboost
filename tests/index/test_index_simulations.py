from pathlib import Path
from typing import Tuple

import pytest

from bamboost import BAMBOOST_LOGGER
from bamboost.core.collection import Collection
from bamboost.index.base import Index


@pytest.fixture
def index_with_collection(tmp_path):
    index = Index(sql_file=":memory:")
    collection_path = tmp_path / "collection"
    collection_path.mkdir()

    index.upsert_collection("COLL001", collection_path)
    return index, collection_path


@pytest.fixture
def index_with_test_collection(test_collection: Collection, tmp_path: Path):
    index = Index(sql_file=":memory:", search_paths=[tmp_path])
    index.upsert_collection(test_collection.uid, test_collection.path)
    return index, test_collection


def test_upsert_simulation(index_with_collection: Tuple[Index, Path]):
    index, collection_path = index_with_collection

    parameters = {"param1": "value1", "param2": 2}
    metadata = {"description": "Test Simulation"}

    index.upsert_simulation(
        collection_uid="COLL001",
        simulation_name="sim1",
        parameters=parameters,
        metadata=metadata,  # pyright: ignore[reportArgumentType]
        collection_path=collection_path,
    )

    sim = index.simulation("COLL001", "sim1")

    assert sim is not None
    assert sim.name == "sim1"
    assert sim.description == "Test Simulation"
    assert sim.parameter_dict == parameters


def test_update_simulation_metadata(
    index_with_test_collection: Tuple[Index, Collection],
):
    index, coll = index_with_test_collection
    sim_name = "testsim1"
    index.upsert_simulation(coll.uid, sim_name)

    new_metadata = {"description": "Updated description"}
    index.update_simulation_metadata(coll.uid, sim_name, new_metadata)  # pyright: ignore[reportArgumentType]

    sim = index.simulation(coll.uid, sim_name)
    assert sim.description == "Updated description"


@pytest.mark.parametrize(
    "params",
    [{"add_1": "new", "add_2": 123}, {"first_name": "tony", "last_name": "stark"}],
)
def test_update_simulation_parameters(
    index_with_test_collection: Tuple[Index, Collection],
    params: dict,
):
    index, coll = index_with_test_collection
    sim_name = "testsim1"
    index.upsert_simulation(coll.uid, sim_name)

    old_params = index.simulation(coll.uid, sim_name).parameter_dict
    index.update_simulation_parameters(coll.uid, sim_name, params)
    sim = index.simulation(coll.uid, sim_name)

    # update old_params for comparison
    old_params.update(params)

    assert sim.parameter_dict == old_params
