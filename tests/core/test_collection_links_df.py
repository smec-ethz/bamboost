import pytest
import pandas as pd
from bamboost.core.collection import Collection
from bamboost.index import SimulationUID


def test_collection_include_links_df(tmp_collection: Collection):
    # Create target collection and simulation
    target_sim = tmp_collection.add(
        name="target1",
        parameters={"p_target": 100, "shared": "target"},
    )

    # Create source simulation with a link
    source_sim = tmp_collection.add(
        name="source1",
        parameters={"p_source": 1, "shared": "source"},
        links={"ref": target_sim.uid},
    )

    # Check default DF (should NOT include linked parameters)
    df = tmp_collection.df
    assert "ref.p_target" not in df.columns
    assert "shared" in df.columns
    assert df.loc[df["name"] == "source1", "shared"].iloc[0] == "source"

    # Check with include_links()
    df_linked = tmp_collection.include_links("ref").df
    assert "ref.p_target" in df_linked.columns
    assert df_linked.loc[df_linked["name"] == "source1", "ref.p_target"].iloc[0] == 100

    # Check prefixing for shared parameters
    assert "ref.shared" in df_linked.columns
    assert (
        df_linked.loc[df_linked["name"] == "source1", "ref.shared"].iloc[0] == "target"
    )
    assert df_linked.loc[df_linked["name"] == "source1", "shared"].iloc[0] == "source"

    # Check with include_links() (all)
    df_all = tmp_collection.include_links().df
    assert "ref.p_target" in df_all.columns
    assert df_all.loc[df_all["name"] == "source1", "ref.p_target"].iloc[0] == 100
