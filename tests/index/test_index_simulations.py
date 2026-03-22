from pathlib import Path
from typing import Tuple

import pytest

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
    metadata = {"description": "Test Simulation", "tags": ["alpha", "beta", "alpha"]}

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
    assert sim.tags == ["alpha", "beta"]
    assert sim.parameter_dict == parameters


def test_update_simulation_metadata(
    index_with_test_collection: Tuple[Index, Collection],
):
    index, coll = index_with_test_collection
    sim_name = "testsim1"
    index.upsert_simulation(coll.uid, sim_name)

    new_metadata = {"description": "Updated description", "tags": ["first", "first"]}
    index.update_simulation_metadata(coll.uid, sim_name, new_metadata)  # pyright: ignore[reportArgumentType]

    sim = index.simulation(coll.uid, sim_name)
    assert sim.description == "Updated description"
    assert sim.tags == ["first"]


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


from bamboost.index.uids import SimulationUID


def test_upsert_simulation_with_links(index_with_collection: Tuple[Index, Path]):
    index, collection_path = index_with_collection

    # Target simulations must exist first
    index.upsert_simulation(
        "COLL001",
        "sim2",
        metadata={},
        parameters={},
        links={},
        collection_path=collection_path,
    )
    index.upsert_simulation(
        "COLL001",
        "sim3",
        metadata={},
        parameters={},
        links={},
        collection_path=collection_path,
    )

    links = {"ref1": "COLL001:sim2", "ref2": SimulationUID("COLL001", "sim3")}

    index.upsert_simulation(
        collection_uid="COLL001",
        simulation_name="sim1",
        metadata={},
        parameters={},
        links=links,
        collection_path=collection_path,
    )

    sim = index.simulation("COLL001", "sim1")
    assert sim is not None
    assert sim.links == {"ref1": "COLL001:sim2", "ref2": "COLL001:sim3"}


def test_upsert_simulation_with_missing_target(
    index_with_collection: Tuple[Index, Path],
):
    index, collection_path = index_with_collection

    links = {"ref1": "COLL001:MISSING_SIM"}

    with pytest.raises(ValueError, match="does not exist"):
        index.upsert_simulation(
            collection_uid="COLL001",
            simulation_name="sim1",
            metadata={},
            parameters={},
            links=links,
            collection_path=collection_path,
        )


def test_update_simulation_links(index_with_collection: Tuple[Index, Path]):
    index, collection_path = index_with_collection

    # Target must exist
    index.upsert_simulation(
        "COLL001",
        "sim2",
        metadata={},
        parameters={},
        links={},
        collection_path=collection_path,
    )

    index.upsert_simulation(
        "COLL001",
        "sim1",
        metadata={},
        parameters={},
        links={},
        collection_path=collection_path,
    )

    links = {"ref1": "COLL001:sim2"}
    index.update_simulation_links("COLL001", "sim1", links)

    sim = index.simulation("COLL001", "sim1")
    assert sim.links == links


def test_collection_links(index_with_collection: Tuple[Index, Path]):
    index, collection_path = index_with_collection

    # Targets
    index.upsert_simulation(
        "COLL001",
        "T1",
        metadata={},
        parameters={},
        links={},
        collection_path=collection_path,
    )
    index.upsert_simulation(
        "COLL001",
        "T2",
        metadata={},
        parameters={},
        links={},
        collection_path=collection_path,
    )

    index.upsert_simulation(
        "COLL001",
        "sim1",
        metadata={},
        parameters={},
        links={"l1": "COLL001:T1"},
        collection_path=collection_path,
    )
    index.upsert_simulation(
        "COLL001",
        "sim2",
        metadata={},
        parameters={},
        links={"l2": "COLL001:T2"},
        collection_path=collection_path,
    )

    coll = index.collection("COLL001")
    assert coll.links == {"sim1": {"l1": "COLL001:T1"}, "sim2": {"l2": "COLL001:T2"}}

    # Test new efficient links queries
    links = index.collection_links("COLL001")
    assert len(links) == 2
    assert {l.name for l in links} == {"l1", "l2"}

    links_map = index.collection_links_map("COLL001")
    assert links_map == {"sim1": {"l1": "COLL001:T1"}, "sim2": {"l2": "COLL001:T2"}}

    # Test backlinks
    backlinks = index.backlinks("COLL001:T1")
    assert len(backlinks) == 1
    assert backlinks[0][0] == SimulationUID("COLL001", "sim1")
    assert backlinks[0][1] == "l1"
