import pytest

from bamboost import Collection


def test_parameter_key_validation_reserved_add(tmp_collection_burn: Collection):
    # 'name' is a reserved key
    with pytest.raises(ValueError, match="is reserved"):
        tmp_collection_burn.add(parameters={"name": "invalid"})


def test_parameter_key_validation_period_add(tmp_collection_burn: Collection):
    # periods are not allowed in keys
    with pytest.raises(ValueError, match="cannot contain a period"):
        tmp_collection_burn.add(parameters={"invalid.key": 1})


def test_parameter_key_validation_reserved_update(tmp_collection_burn: Collection):
    sim = tmp_collection_burn.add("sim1", parameters={"a": 1})
    with sim.edit() as sw:
        with pytest.raises(ValueError, match="is reserved"):
            sw.parameters["status"] = "finished"

        with pytest.raises(ValueError, match="is reserved"):
            sw.parameters.update({"tags": ["a"]})


def test_parameter_key_validation_period_update(tmp_collection_burn: Collection):
    sim = tmp_collection_burn.add("sim1", parameters={"a": 1})
    with sim.edit() as sw:
        # Note: sw.parameters["b.c"] = 1 is actually supported by the code
        # (it creates a nested dict), but the request said keys should NEVER
        # contain a period.
        # Actually, Parameters.__setitem__ validates the key BEFORE splitting it.
        with pytest.raises(ValueError, match="cannot contain a period"):
            sw.parameters["b.c"] = 1

        with pytest.raises(ValueError, match="cannot contain a period"):
            sw.parameters.update({"x.y": 2})


def test_parameter_key_validation_links_reserved(tmp_collection_burn: Collection):
    with pytest.raises(ValueError, match="is reserved"):
        tmp_collection_burn.add(parameters={"links": "something"})
