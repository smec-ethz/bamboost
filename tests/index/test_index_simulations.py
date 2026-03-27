from pathlib import Path
from typing import Tuple

import pytest

from bamboost.core.collection import Collection
from bamboost.exceptions import InvalidSimulationUIDError
from bamboost.index import Index, SimulationUID


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

    sim = index.simulation(SimulationUID("COLL001", "sim1"))

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

    sim = index.simulation(SimulationUID(coll.uid, sim_name))
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

    old_params = index.simulation(SimulationUID(coll.uid, sim_name)).parameter_dict
    index.update_simulation_parameters(coll.uid, sim_name, params)
    sim = index.simulation(SimulationUID(coll.uid, sim_name))

    # update old_params for comparison
    old_params.update(params)

    assert sim.parameter_dict == old_params


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

    links = {
        "ref1": SimulationUID("COLL001:sim2"),
        "ref2": SimulationUID("COLL001", "sim3"),
    }

    index.upsert_simulation(
        collection_uid="COLL001",
        simulation_name="sim1",
        metadata={},
        parameters={},
        links=links,
        collection_path=collection_path,
    )

    sim = index.simulation(SimulationUID("COLL001", "sim1"))
    assert sim is not None
    assert sim.links == links


def test_upsert_simulation_with_missing_target(
    index_with_collection: Tuple[Index, Path],
):
    index, collection_path = index_with_collection

    links = {"ref1": "COLL001:MISSING_SIM"}

    with (
        pytest.MonkeyPatch.context() as m,
        pytest.raises(InvalidSimulationUIDError, match="cannot be found"),
    ):
        # monkeypatch sync_collection to avoid syncing, we only check the existing index
        m.setattr(Index, "sync_collection", lambda self, collection_uid: None)

        index.upsert_simulation(
            collection_uid="COLL001",
            simulation_name="sim1",
            metadata={},
            parameters={},
            collection_path=collection_path,
        )

        # this line should raise the error because the target simulation does not exist in the index
        index.update_simulation_links(
            "COLL001", "sim1", links, raise_on_invalid_target=True
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

    links = {"ref1": SimulationUID("COLL001:sim2")}
    index.update_simulation_links("COLL001", "sim1", links)

    sim = index.simulation(SimulationUID("COLL001", "sim1"))
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
    expected_links = {
        "sim1": {"l1": SimulationUID("COLL001:T1")},
        "sim2": {"l2": SimulationUID("COLL001:T2")},
    }
    assert coll.links == expected_links

    # Test new efficient links queries
    links = index.collection_links("COLL001")
    assert len(links) == 2
    assert {l.name for l in links} == {"l1", "l2"}

    links_map = index.collection_links_map("COLL001")
    assert links_map == expected_links

    # Test backlinks
    backlinks = index.backlinks("COLL001:T1")
    assert len(backlinks) == 1
    assert backlinks[0][0] == SimulationUID("COLL001", "sim1")
    assert backlinks[0][1] == "l1"


def test_stale_link_storage_and_healing(index_with_collection: Tuple[Index, Path]):
    index, collection_path = index_with_collection
    index.search_paths.add(collection_path.parent)

    # Create identifier file so find_collection works
    from bamboost.index import create_identifier_file
    create_identifier_file(collection_path, "COLL001")

    # Create dummy simulation file
    import h5py
    sim1_path = collection_path / "sim1"
    sim1_path.mkdir()
    with h5py.File(sim1_path / "data.h5", "w") as f:
        f.create_group(".parameters")
        f.create_group(".links")

    # Create sim1 with a link to a non-existent sim2
    links = {"ref1": "COLL001:sim2"}
    with index.sql_transaction():
        index.upsert_simulation(
            collection_uid="COLL001",
            simulation_name="sim1",
            metadata={},
            parameters={},
            links=links,
            collection_path=collection_path,
        )

    # Verify link is stored as stale (SimulationUID is present in JSON but NOT in relational table)
    sim1 = index.simulation(SimulationUID("COLL001", "sim1"))
    assert sim1 is not None
    assert sim1.links == {"ref1": SimulationUID("COLL001", "sim2")}

    # Check database directly
    from sqlalchemy import select
    from bamboost.index import store

    with index.sql_transaction():
        # 1. Should be in JSON column
        row = index._s.execute(
            select(store.simulations_table).where(
                store.simulations_table.c.name == "sim1"
            )
        ).mappings().first()
        assert row["links"] == {"ref1": "COLL001:sim2"}

        # 2. Should NOT be in relational table yet (target not found)
        rel_row = index._s.execute(
            select(store.simulation_links_table).where(
                store.simulation_links_table.c.name == "ref1"
            )
        ).mappings().first()
        assert rel_row is None

    # Now add the target sim2
    sim2_path = collection_path / "sim2"
    sim2_path.mkdir()
    with h5py.File(sim2_path / "data.h5", "w") as f:
        f.create_group(".parameters")
        f.create_group(".links")

    with index.sql_transaction():
        index.upsert_simulation(
            "COLL001",
            "sim2",
            metadata={},
            parameters={},
            links={},
            collection_path=collection_path,
        )

    # Healing should happen during sync
    index.sync_collection("COLL001")

    # Verify target_id is now populated in relational table
    with index.sql_transaction():
        row = index._s.execute(
            select(store.simulation_links_table).where(
                store.simulation_links_table.c.name == "ref1"
            )
        ).mappings().first()
        assert row is not None
        assert row["target_id"] is not None
