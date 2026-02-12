"""Tests for bamboost.exceptions module."""

import pytest

from bamboost.exceptions import DuplicateSimulationError, InvalidCollectionError


class TestInvalidCollectionError:
    """Tests for InvalidCollectionError exception."""

    def test_invalid_collection_error_is_os_error(self):
        """Test that InvalidCollectionError is a subclass of OSError."""
        assert issubclass(InvalidCollectionError, OSError)

    def test_invalid_collection_error_can_be_raised(self):
        """Test that InvalidCollectionError can be raised."""
        with pytest.raises(InvalidCollectionError):
            raise InvalidCollectionError("Test message")

    def test_invalid_collection_error_with_message(self):
        """Test that InvalidCollectionError preserves the error message."""
        message = "Collection not found at path /foo/bar"
        with pytest.raises(InvalidCollectionError, match=message):
            raise InvalidCollectionError(message)


class TestDuplicateSimulationError:
    """Tests for DuplicateSimulationError exception."""

    def test_duplicate_simulation_error_is_value_error(self):
        """Test that DuplicateSimulationError is a subclass of ValueError."""
        assert issubclass(DuplicateSimulationError, ValueError)

    def test_duplicate_simulation_error_initialization(self):
        """Test that DuplicateSimulationError initializes with duplicates tuple."""
        duplicates = ("sim1", "sim2", "sim3")
        error = DuplicateSimulationError(duplicates)
        assert error.duplicates == duplicates

    def test_duplicate_simulation_error_can_be_raised(self):
        """Test that DuplicateSimulationError can be raised."""
        duplicates = ("sim1",)
        with pytest.raises(DuplicateSimulationError):
            raise DuplicateSimulationError(duplicates)

    def test_duplicate_simulation_error_str_representation(self):
        """Test the string representation of DuplicateSimulationError."""
        duplicates = ("sim1", "sim2")
        error = DuplicateSimulationError(duplicates)
        error_str = str(error)
        assert "Duplicates:" in error_str
        assert "('sim1', 'sim2')" in error_str

    def test_duplicate_simulation_error_empty_duplicates(self):
        """Test DuplicateSimulationError with empty duplicates tuple."""
        duplicates = tuple()
        error = DuplicateSimulationError(duplicates)
        assert error.duplicates == tuple()
        assert "Duplicates: ()" in str(error)

    def test_duplicate_simulation_error_single_duplicate(self):
        """Test DuplicateSimulationError with a single duplicate."""
        duplicates = ("single_sim",)
        error = DuplicateSimulationError(duplicates)
        assert error.duplicates == duplicates
        assert "single_sim" in str(error)
