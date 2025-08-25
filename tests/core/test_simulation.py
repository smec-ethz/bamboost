import os
from datetime import datetime
from pathlib import Path
from typing import Type
from unittest.mock import ANY, MagicMock, patch

import pytest

from bamboost import constants
from bamboost.core.collection import Collection
from bamboost.core.hdf5.attrsdict import AttrsDict
from bamboost.core.hdf5.file import FileMode
from bamboost.core.hdf5.ref import Group
from bamboost.core.simulation.base import (
    Simulation,
    SimulationName,
    SimulationWriter,
    Status,
    StatusInfo,
    _Simulation,
)
from bamboost.core.simulation.series import Series


@pytest.fixture
def tmp_simulation(tmp_path: Path) -> Simulation:
    # create collection
    Collection(tmp_path)

    # create a directory for the simulation (simulates coll.add)
    tmp_path.joinpath("test").mkdir()
    sim = SimulationWriter("test", tmp_path)
    # create the file
    sim.initialize()

    return Simulation("test", tmp_path)


@pytest.fixture
def tmp_simulation_writer(tmp_path: Path) -> SimulationWriter:
    # create collection
    Collection(tmp_path)

    # create a directory for the simulation (simulates coll.add)
    tmp_path.joinpath("test").mkdir()
    sim = SimulationWriter("test", tmp_path)
    # create the file
    sim.initialize()

    return sim


@pytest.fixture
def mock_hdf5():
    """Mock the HDF5 file interactions."""
    with patch("bamboost.core.hdf5.file.HDF5File", autospec=True) as mock_hdf5:
        yield mock_hdf5


def test_import():
    assert Simulation
    assert SimulationWriter


def test_update_database(tmp_collection: Collection):
    sim = tmp_collection.add("test_update_database")
    index = tmp_collection._index

    new_metadata = {"ignored": "ignored", "status": "test", "description": "test"}
    sim.update_database(metadata=new_metadata)
    assert index.simulation(tmp_collection.uid, sim.name).status == "test"
    assert index.simulation(tmp_collection.uid, sim.name).description == "test"

    new_parameters = {"last_name": "Bourne", "first_name": "Jane"}
    sim.update_database(parameters=new_parameters)
    assert (
        index.simulation(tmp_collection.uid, sim.name).parameter_dict == new_parameters
    )


@pytest.mark.parametrize(
    "status_string, expected",
    [
        ("initialized", Status.INITIALIZED),
        ("started", Status.STARTED),
        ("finished", Status.FINISHED),
        ("failed", Status.FAILED),
        ("some random stuff", Status.UNKNOWN),
        ("", Status.UNKNOWN),
    ],
)
def test_status_info_parse(status_string, expected):
    status_info = StatusInfo.parse(status_string)

    assert status_info.status == expected


def test_status_info_message():
    status_string = "failed [RuntimeError - 'Some error']"
    status_info = StatusInfo.parse(status_string)
    assert status_info.status == Status.FAILED
    assert status_info.message == "RuntimeError - 'Some error'"


def test_simulation_name_given():
    name = SimulationName("test")
    assert name == "test"


def test_simulation_name_generation():
    length = 10
    name = SimulationName(length=length)
    assert name is not None
    assert len(name) == length


def test_init(tmp_collection: Collection):
    params = {"test": "test"}
    tmp_collection.add("test", parameters=params)

    sim = Simulation("test", tmp_collection.path)
    assert sim.name == "test"
    assert sim.path == tmp_collection.path.joinpath("test")
    assert sim._index  # reference to index database exists
    assert sim.collection_uid == tmp_collection.uid


def test_fail_non_existing(tmp_collection_burn: Collection):
    """Test that init a non-existing simulation raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        Simulation("test", tmp_collection_burn.path)


def test_html_repr(tmp_collection_burn: Collection):
    sim = tmp_collection_burn.add("test")
    assert sim._repr_html_() is not None


def test_root_group(test_collection: Collection):
    sim = test_collection["testsim1"]
    assert isinstance(sim.root, Group)


@pytest.mark.parametrize(
    "cls, mutable", [(Simulation, False), (SimulationWriter, True)]
)
def test_mutability(test_collection: Collection, cls: Type[_Simulation], mutable: bool):
    path = test_collection.path
    sim = cls("testsim1", path)
    assert sim.mutable == mutable


def test_from_uid(test_collection: Collection):
    uid = test_collection.uid
    sim_uid = f"{uid}{constants.UID_SEPARATOR}testsim1"
    sim = Simulation.from_uid(sim_uid)
    sim2 = Simulation("testsim1", test_collection.path)
    assert sim == sim2


def test_get_writer_from_simulation(test_collection: Collection):
    sim = test_collection["testsim1"]
    writer = sim.edit()
    assert isinstance(writer, SimulationWriter)


@pytest.mark.skip("Dumb test")
def test_parameters(test_collection: Collection):
    sim = test_collection["testsim1"]
    assert isinstance(sim.parameters, AttrsDict)


def test_status(test_collection: Collection):
    sim = test_collection["testsim1"]
    assert sim.status == StatusInfo(Status.INITIALIZED)


def test_created_at_is_datetime(test_collection: Collection):
    sim = test_collection["testsim1"]
    assert isinstance(sim.created_at, datetime)


def test_simulation_enter_path(tmp_simulation: Simulation):
    initial_dir = Path.cwd()

    with tmp_simulation.enter_path():
        assert Path.cwd() == tmp_simulation.path

    assert Path.cwd() == initial_dir


def test_create_xdmf(tmp_simulation):
    """Test that create_xdmf correctly calls XDMFWriter and writes an XDMF file."""
    tmp_simulation = tmp_simulation.edit()

    with (
        patch("bamboost.core.simulation.xdmf.XDMFWriter") as mock_xdmf_writer,
        patch.object(tmp_simulation, "meshes") as mock_group_meshes,
    ):
        mock_group_meshes.__getitem__.return_value = "test_mesh"
        mock_writer_instance = mock_xdmf_writer.return_value
        mock_writer_instance.write_file = MagicMock()

        field_names = ["velocity", "pressure"]
        timesteps = [0.0, 0.1, 0.2]
        filename = tmp_simulation.path / "test_output.xdmf"
        mesh_name = "test_mesh"

        tmp_simulation.create_xdmf(
            field_names=field_names,
            timesteps=timesteps,
            filename=filename,
            mesh_name=mesh_name,
        )

        # Ensure XDMFWriter was instantiated with the correct file
        mock_xdmf_writer.assert_called_once_with(tmp_simulation._file)

        # Ensure methods were called with expected arguments
        mock_writer_instance.add_mesh.assert_called_once_with("test_mesh")
        mock_writer_instance.add_timeseries.assert_called_once_with(
            timesteps, tmp_simulation.data.get_fields(*field_names), mesh_name
        )
        mock_writer_instance.write_file.assert_called_once_with(filename)


# SimulationWriter tests
def test_writer_context_success(tmp_simulation: Simulation):
    writer = tmp_simulation.edit()

    with writer as w:
        assert w.status.status == Status.STARTED

    assert writer.status.status == Status.FINISHED


def test_writer_context_failure(tmp_simulation: Simulation):
    writer = tmp_simulation.edit()

    with pytest.raises(ValueError):
        with writer as w:
            assert w.status.status == Status.STARTED
            raise ValueError("test")

    assert writer.status.status == Status.FAILED


def test_simulation_writer_set_status(tmp_simulation_writer):
    """Test that setting the status updates metadata correctly."""
    tmp_simulation_writer.status = StatusInfo(Status.STARTED)
    assert tmp_simulation_writer.metadata["status"] == "started"

    tmp_simulation_writer.status = StatusInfo(Status.FAILED, "An error occurred")
    assert tmp_simulation_writer.metadata["status"] == "failed [An error occurred]"


def test_simulation_writer_set_description(tmp_simulation_writer):
    """Test that setting the description updates metadata correctly."""
    tmp_simulation_writer.description = "This is a test description"
    assert tmp_simulation_writer.metadata["description"] == "This is a test description"


def test_simulation_writer_require_series(tmp_simulation_writer):
    """Test that require_series initializes the series in the HDF5 file if missing."""
    with patch.object(
        tmp_simulation_writer,
        "_initialize_series",
        wraps=tmp_simulation_writer._initialize_series,
    ) as mock_init_series:
        series = tmp_simulation_writer.require_series("custom_series")

        # Ensure _initialize_series is called since the series was missing
        mock_init_series.assert_called_once_with("custom_series")

        # Ensure the correct object is returned
        assert isinstance(series, Series)


def test_simulation_writer_copy_files(tmp_simulation_writer, tmp_path):
    """Test that copy_files correctly copies files and directories to the simulation folder."""
    source_file = tmp_path / "test_file.txt"
    source_file.write_text("Test content")

    source_dir = tmp_path / "test_dir"
    source_dir.mkdir()
    (source_dir / "file_in_dir.txt").write_text("Inside directory")

    with patch("shutil.copy") as mock_copy, patch("shutil.copytree") as mock_copytree:
        tmp_simulation_writer.copy_files([source_file, source_dir])

        # Ensure shutil.copy was called for the file
        mock_copy.assert_called_once_with(source_file, tmp_simulation_writer.path)

        # Ensure shutil.copytree was called for the directory
        mock_copytree.assert_called_once_with(source_dir, tmp_simulation_writer.path)


def test_simulation_writer_create_run_script(tmp_simulation_writer):
    """Test that create_run_script correctly generates a bash script."""
    script_path = tmp_simulation_writer.path / constants.RUN_FILE_NAME
    commands = ["echo 'Hello'", "echo 'World'"]
    sbatch_kwargs = {"--time": "02:00:00", "--ntasks": "4"}

    # Test with euler=True (should include sbatch options)
    tmp_simulation_writer.create_run_script(
        commands, euler=True, sbatch_kwargs=sbatch_kwargs
    )

    assert script_path.exists()

    with script_path.open() as f:
        content = f.read()

    # Ensure sbatch options are included
    assert "#SBATCH --time=02:00:00" in content
    assert "#SBATCH --ntasks=4" in content
    assert (
        f"#SBATCH --output={tmp_simulation_writer.path.joinpath(tmp_simulation_writer.name + '.out')}"
        in content
    )
    assert f"#SBATCH --job-name={tmp_simulation_writer.uid}" in content

    # Ensure commands are included
    assert "echo 'Hello'" in content
    assert "echo 'World'" in content

    # Ensure environment variables are set
    assert f"SIMULATION_DIR={tmp_simulation_writer.path.as_posix()}" in content
    assert f"SIMULATION_ID={tmp_simulation_writer.uid}" in content

    # Ensure metadata is updated
    assert tmp_simulation_writer.metadata["submitted"] is False

    # Test with euler=False (should NOT include sbatch options)
    tmp_simulation_writer.create_run_script(commands, euler=False)

    with script_path.open() as f:
        content_no_sbatch = f.read()

    assert "#SBATCH" not in content_no_sbatch  # No sbatch directives
    assert "echo 'Hello'" in content_no_sbatch
    assert "echo 'World'" in content_no_sbatch


def test_simulation_writer_run_simulation(tmp_simulation_writer: SimulationWriter):
    """Test that run_simulation correctly executes the run script."""
    # Ensure the script exists
    script_path = tmp_simulation_writer._bash_file
    script_path.touch()  # Create an empty file

    with (
        patch("subprocess.run") as mock_run,
        patch.dict(os.environ, {"BAMBOOST_MPI": "1"}, clear=False),
    ):
        tmp_simulation_writer.run_simulation(executable="bash")

        # Ensure subprocess.run is called with the correct arguments
        mock_run.assert_called_once_with(["bash", script_path.as_posix()], env=ANY)

        # Ensure BAMBOOST_MPI is removed from environment
        _, kwargs = mock_run.call_args
        assert "BAMBOOST_MPI" not in kwargs["env"]

        # Ensure metadata is updated
        assert tmp_simulation_writer.metadata["submitted"] is True


def test_simulation_writer_run_simulation_missing_script(tmp_simulation_writer):
    """Test that run_simulation raises an error if the script is missing."""
    with pytest.raises(FileNotFoundError, match="Run script .* does not exist"):
        tmp_simulation_writer.run_simulation(executable="bash")


def test_simulation_writer_submit_simulation(tmp_simulation_writer):
    """Test that submit_simulation correctly calls sbatch."""
    with patch.object(tmp_simulation_writer, "run_simulation") as mock_run_simulation:
        tmp_simulation_writer.submit_simulation()

        # Ensure run_simulation is called with sbatch
        mock_run_simulation.assert_called_once_with(executable="sbatch")
