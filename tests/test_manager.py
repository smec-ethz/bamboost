# test_manager.py

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from bamboost import Manager, Simulation, SimulationWriter, index


@pytest.fixture()
def temp_manager():
    temp_dir = tempfile.mkdtemp()
    db = Manager(path=temp_dir)
    yield db
    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        pass


def test_if_new_manager_created():
    temp_dir = tempfile.mkdtemp()
    db = Manager(temp_dir)
    assert os.path.isdir(temp_dir)


class TestManager:
    def test_create_simulation_creates_new_file(self, temp_manager: Manager):
        sim_uid = "this"
        writer = temp_manager.create_simulation(uid=sim_uid)
        h5_file = os.path.join(temp_manager.path, sim_uid, f"{sim_uid}.h5")
        assert os.path.isfile(h5_file)

    @pytest.mark.parametrize("prefix", ["dummy"])
    def test_create_simulation_argument_prefix(self, temp_manager: Manager, prefix):
        sim_uid = "this"
        writer = temp_manager.create_simulation(uid=sim_uid, prefix=prefix)

        assert writer.uid == f"{prefix}_this"

    def test_create_simulation_argument_note(self, temp_manager: Manager):
        sim_uid = "this"
        writer = temp_manager.create_simulation(uid=sim_uid, note="dummy note")

        assert writer.metadata["notes"] == "dummy note"

    def test_create_simulation_argument_files(self, temp_manager: Manager):
        sim_uid = "this"
        # Create Files
        with open(os.path.join(temp_manager.path, "a.txt"), "w") as f:
            f.write("a")
        with open(os.path.join(temp_manager.path, "b.txt"), "w") as f:
            f.write("b")

        files = [f"{temp_manager.path}/a.txt", f"{temp_manager.path}/b.txt"]
        writer = temp_manager.create_simulation(uid=sim_uid, files=files)

        assert all([os.path.isfile(os.path.join(writer.path, f)) for f in files])

    def test_create_simulation_argument_links(self, temp_manager: Manager):
        sim_uid = "this"
        linked_sim = temp_manager.create_simulation(uid="link_to_this")
        writer = temp_manager.create_simulation(
            uid=sim_uid, links={"mesh": linked_sim.get_full_uid()}
        )

        assert writer.links["mesh"].metadata == linked_sim.metadata

    @pytest.mark.parametrize("fix_df", [True, False])
    def test_manager_length(self, temp_manager: Manager, fix_df: bool):
        temp_manager.FIX_DF = fix_df
        temp_manager.create_simulation()
        temp_manager.create_simulation()
        temp_manager.create_simulation()
        assert len(temp_manager) == 3

    def test_get_single_simulation_base_case(self, temp_manager: Manager):
        sim_uid = "this"
        temp_manager.create_simulation()
        temp_manager.create_simulation(sim_uid)
        db = Manager(temp_manager.path)
        sim = db.sim(sim_uid)
        assert isinstance(sim, Simulation)
        assert sim.path == os.path.join(temp_manager.path, sim_uid)


class TestManagerDuplicate:
    def test_prompt(self, monkeypatch, temp_manager: Manager):
        def patchinput(*args, **kwargs):
            raise RuntimeError("input called")

        monkeypatch.setattr("builtins.input", patchinput)
        with pytest.raises(RuntimeError, match="input called"):
            params = dict(a=1, b=2)
            temp_manager.create_simulation(parameters=params)
            sim = temp_manager.create_simulation(
                parameters=params, duplicate_action="prompt"
            )

        # Check that there is no prompt here
        params = dict(a=1, b=2)
        sim = temp_manager.create_simulation(parameters=params, duplicate_action="a")

    def test_abort(self, temp_manager: Manager):
        params = dict(a=1, b=2)
        temp_manager.create_simulation(parameters=params)
        sim = temp_manager.create_simulation(parameters=params, duplicate_action="a")
        assert sim is None
        assert len(temp_manager) == 1

    def test_create_new(self, temp_manager: Manager):
        params = dict(a=1, b=2)
        temp_manager.create_simulation(parameters=params)
        sim = temp_manager.create_simulation(parameters=params, duplicate_action="c")
        assert isinstance(sim, SimulationWriter)
        assert len(temp_manager) == 2


# --------------------------------------------------------------------------
# Database syncing between SQLite and Filesystem
# --------------------------------------------------------------------------
def test_if_table_created(temp_manager: Manager):
    uid = temp_manager.UID
    table_name = f"db_{uid}"
    res = index.IndexAPI().fetch("SELECT name FROM sqlite_master WHERE type='table';")
    # assert that the database table exists
    assert (table_name,) in res
    # assert that the table with update times exists
    assert (f"db_{uid}_t",) in res


def test_dataframe_integrity(temp_manager: Manager):
    _index = index.IndexAPI()  # shortcutting calling IndexAPI() every time
    booleans = {
        "1": True,
        "2": False,
        "3": True,
    }

    @_index.commit_once
    def create_sims(booleans: list):
        for args in zip([1, 2, 3], booleans, ["a", "b", "c"]):
            sim = temp_manager.create_simulation(
                uid=args[1],
                parameters={
                    "int": args[0],
                    "float": 1.0,
                    "str": args[2],
                    "boolean": booleans[args[1]],
                    "boolean2": False,
                    "array": np.array([1, 2, 3]),
                },
                skip_duplicate_check=True,
            )

    create_sims(booleans)

    # delete the table to force re-creation -> booleans are wrong!!!
    with _index.open():
        _index._cursor.execute(f"DROP TABLE db_{temp_manager.UID};")
        _index._cursor.execute(f"DROP TABLE db_{temp_manager.UID}_t;")
        _index.commit()

    db = Manager(uid=temp_manager.UID)
    df = db.df.set_index("id")

    for uid, val in booleans.items():
        assert df.loc[uid, "boolean"] == val


def test_duplicates_detect(temp_manager: Manager):
    db = temp_manager
    params1 = dict(a=1, b=2)
    db.create_simulation("1", params1)
    db.create_simulation("2", dict(a=4, b=5))

    assert db._list_duplicates(params1)[0] == "1"

    sim = db.create_simulation(parameters=params1, duplicate_action="c")

    assert {"1", sim.uid} == set(db._list_duplicates(params1))


def test_duplicates_new_key(temp_manager: Manager):
    db = temp_manager
    params1 = dict(a=1, b=2)
    db.create_simulation("1", params1)

    params2 = dict(a=1, b=2, c=3)

    assert len(db._list_duplicates(params2)) == 0


def test_duplicates_handles_nan(temp_manager: Manager):
    db = temp_manager
    params1 = dict(a=1, b=2)
    db.create_simulation("1", params1)

    params2 = dict(a=1, b=2, c=3)
    db.create_simulation("2", params2)
    # This introduces a nan entry in simulation "1"
    assert np.isnan(db.df[db.df.id == "1"].c.iloc[0])

    params3 = dict(a=1, b=2, c=4)
    # Simulation 2 shall not be regarded as duplicate of simulation 1, because of the NaN
    assert len(db._list_duplicates(params3)) == 0


def test_duplicates_lists(temp_manager: Manager):
    db = temp_manager
    params1 = dict(a=[1, 2, 3])
    db.create_simulation("1", params1)

    params2 = dict(a=[2, 3, 4])
    assert len(db._list_duplicates(params2)) == 0

    db.create_simulation("2", params2)

    assert len(db._list_duplicates(params2)) == 1
    assert len(db._list_duplicates(params1)) == 1

    # Works with arrays
    params2 = dict(a=np.array([2, 3, 4]))
    assert len(db._list_duplicates(params2)) == 1
    db.create_simulation("3", params2, duplicate_action="c")
    assert len(db._list_duplicates(params2)) == 2

    # different length -> dnot duplicates
    # Works with arrays
    params2 = dict(a=np.array([2, 3, 4, 5]))
    assert len(db._list_duplicates(params2)) == 0

    # 2D arrays
    params2 = dict(a=np.array([[2, 3], [4, 5]]))
    assert len(db._list_duplicates(params2)) == 0
    db.create_simulation(parameters=params2, duplicate_action="c")
    assert len(db._list_duplicates(params2)) == 1

    # We do not plan to support sets


def test_replace_duplicates(temp_manager: Manager):
    db = temp_manager
    params1 = dict(a=[1, 2, 3])
    db.create_simulation("1", parameters=params1)
    assert len(db.df) == 1

    db.create_simulation("2", parameters=params1, duplicate_action="r")
    assert len(db.df) == 1
    assert db._get_uids()[0] == "2"


def test_altered_uid(temp_manager: Manager):
    db = temp_manager
    params1 = dict(a=[1, 2, 3])
    db.create_simulation("1", parameters=params1)
    assert len(db.df) == 1

    db.create_simulation(parameters=params1, duplicate_action="c")
    db.get_view()

    assert len(db.df) == 2

    assert set(db.df["id"]) == {"1", "1.1"}


@pytest.mark.parametrize(
    "params,expected",
    [
        (dict(a=1, b=2), ("1", "2", "3")),
        (dict(a=1, b=2, c=3), ("2",)),
        (dict(c=lambda c: c < 4), ("2",)),
    ],
)
def test_find(temp_manager: Manager, params: dict, expected: tuple):
    db = temp_manager
    # mock the database dataframe from sql
    db._dataframe = pd.DataFrame.from_records(
        [
            dict(id="1", a=1, b=2),
            dict(id="2", a=1, b=2, c=3),
            dict(id="3", a=1, b=2, c=4),
            dict(id="4", a=1, b=10),
        ]
    )

    assert isinstance(db.find(params), pd.DataFrame)
    assert set(db.find(params).id) == set(expected)
