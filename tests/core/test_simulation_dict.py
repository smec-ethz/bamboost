from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bamboost import constants
from bamboost.core.hdf5.file import FileMode
from bamboost.core.simulation.dict import Links, Metadata, Parameters


@pytest.fixture
def mock_simulation():
    """Fixture for a mock simulation object."""
    sim = MagicMock()
    sim._file.open = MagicMock()
    sim.update_database = MagicMock()
    return sim


@pytest.fixture
def mock_hdf5_group():
    """Fixture for mocking an HDF5 group."""
    mock_group = MagicMock()
    mock_group.attrs = {}
    return mock_group


@pytest.fixture
def parameters(mock_simulation, mock_hdf5_group):
    """Fixture for initializing a Parameters instance."""
    with patch(
        "bamboost.core.hdf5.file.HDF5File.__getitem__", return_value=mock_hdf5_group
    ):
        return Parameters(mock_simulation)


def test_parameters_read(parameters, mock_hdf5_group):
    """Test that parameters.read() retrieves values from HDF5."""
    mock_hdf5_group.attrs = {"param1": 10, "param2": "value"}
    mock_hdf5_group.items.return_value = []  # No datasets
    with patch.object(parameters._file, "__getitem__", return_value=mock_hdf5_group):
        result = parameters.read()
        assert result == {"param1": 10, "param2": "value"}


def test_parameters_setitem(parameters):
    """Test that setting a parameter updates HDF5 and SQL."""
    parameters["param1"] = 42

    # Ensure the key is stored correctly
    assert parameters._dict["param1"] == 42

    # Ensure update_database is called
    parameters._simulation.update_database.assert_called_once_with(
        parameters={"param1": 42}
    )


def test_parameters_setitem_numpy_array(parameters, mock_hdf5_group):
    """Test that setting a NumPy array stores it as a dataset."""
    mock_hdf5_group.__contains__.return_value = False  # Simulate missing dataset
    array_data = np.array([1, 2, 3])
    parameters["array_key"] = array_data

    # Ensure the key is stored correctly
    assert np.array_equal(parameters._dict["array_key"], array_data)


@pytest.mark.skip
def test_parameters_update(parameters, mock_hdf5_group):
    """Test that update() modifies multiple values in HDF5 and SQL."""
    update_data = {"key1": 100, "key2": "new_value", "key3": np.array([1, 2, 3])}

    with patch.object(mock_hdf5_group, "create_dataset") as mock_create_dataset:
        parameters.update(update_data)

        # Ensure update_database is called
        parameters._simulation.update_database.assert_called_once_with(
            parameters=update_data
        )

        # Ensure correct keys are updated
        assert mock_hdf5_group.attrs["key1"] == 100
        assert mock_hdf5_group.attrs["key2"] == "new_value"

        # Ensure dataset was created for the NumPy array
        mock_create_dataset.assert_called_once_with("key3", data=np.array([1, 2, 3]))


def test_parameters_getitem(parameters):
    """Test that getting a parameter correctly retrieves nested values."""
    parameters._dict = {"nested": {"key": 50}}
    assert parameters["nested.key"] == 50


def test_metadata_setitem():
    """Test that setting a metadata value updates HDF5 and SQL."""
    mock_simulation = MagicMock()
    mock_metadata = Metadata(mock_simulation)

    mock_metadata["created_at"] = "2025-03-12"

    # Ensure HDF5 is updated
    assert mock_metadata["created_at"] == "2025-03-12"

    # Ensure database update is triggered
    mock_simulation.update_database.assert_called_once_with(
        metadata={"created_at": "2025-03-12"}
    )


def test_metadata_update():
    """Test that metadata.update() modifies multiple values in HDF5 and SQL."""
    mock_simulation = MagicMock()
    mock_metadata = Metadata(mock_simulation)

    update_data = {"status": "running", "description": "test simulation"}
    mock_metadata.update(update_data)

    # Ensure HDF5 is updated
    assert mock_metadata["status"] == "running"
    assert mock_metadata["description"] == "test simulation"

    # Ensure database update is triggered
    mock_simulation.update_database.assert_called_once_with(metadata=update_data)


def test_links_initialization(mock_simulation):
    """Test that Links class initializes correctly."""
    links = Links(mock_simulation)
    assert links._path == constants.PATH_LINKS
