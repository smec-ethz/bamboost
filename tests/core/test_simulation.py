from bamboost.core.collection import Collection
from bamboost.core.simulation.base import Simulation, SimulationWriter


def test_import():
    assert Simulation
    assert SimulationWriter


def test_update_database(tmp_collection: Collection):
    sim = tmp_collection.create_simulation("test_update_database")
    index = tmp_collection._index

    new_metadata = {"ignored": "ignored", "status": "test", "description": "test"}
    sim.update_database(metadata=new_metadata)
    assert index.simulation(tmp_collection.uid, sim.name).status == "test"
    assert index.simulation(tmp_collection.uid, sim.name).description == "test"

    new_parameters = {"last_name": "Bourne", "first_name": "Jane"}
    sim.update_database(parameters=new_parameters)
    assert index.simulation(tmp_collection.uid, sim.name).parameter_dict == new_parameters
