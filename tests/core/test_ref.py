from unittest.mock import patch

import h5py
import numpy as np
import pytest

from bamboost._typing import Mutable
from bamboost.core.hdf5.file import HDF5File
from bamboost.core.hdf5.ref import Dataset, Group, H5Reference, RefStatus

DATASET1_DATA = np.array([1, 2, 3])
DATASET2_DATA = np.array([4.2, 5.1, 6.0])


@pytest.fixture
def hdf5_file(tmp_path):
    file = HDF5File(tmp_path.joinpath("test.h5"), mutable=True)
    # create an empty file
    with file.open("w"):
        pass
    return file


@pytest.fixture(scope="module")
def hdf5_file_populated(tmp_path_module):
    file = HDF5File(tmp_path_module.joinpath("test.h5"), mutable=True)
    # add some groups and datasets using h5py directly
    # lenght: 5
    with file.open("w") as f:
        f.create_group("group1")
        grp2 = f.create_group("group2")
        f.create_dataset("dataset1", data=DATASET1_DATA)
        grp2.create_dataset("dataset2", data=DATASET2_DATA)
        grp2.create_group("group3")

    file.file_map.invalidate()

    return file


def test_h5ref_init(hdf5_file):
    ref = H5Reference("/path/to/object", hdf5_file)
    assert ref._path == "/path/to/object"
    assert ref._file == hdf5_file


def test_h5ref_obj(hdf5_file_populated: HDF5File):
    ref = H5Reference("/group1", hdf5_file_populated)
    with ref.open("r"):
        assert isinstance(ref._obj, h5py.Group)
        assert ref._status == RefStatus.VALID


def test_h5ref_obj_error_file_not_open(hdf5_file_populated: HDF5File):
    ref = H5Reference("/group1", hdf5_file_populated)
    with pytest.raises(KeyError):
        obj = ref._obj


def test_h5ref_obj_error_invalid_path(hdf5_file_populated: HDF5File):
    ref = H5Reference("/group1/invalid", hdf5_file_populated)
    with ref.open("r"):
        with pytest.raises(KeyError):
            obj = ref._obj


def test_h5ref_repr(hdf5_file):
    ref = H5Reference("/path/to/object", hdf5_file)
    assert "HDF5 H5Reference" in repr(ref)


def test_h5reference_new(hdf5_file_populated):
    # Test creating a new Group
    group = H5Reference.new("/group1", hdf5_file_populated)
    assert isinstance(group, Group)

    # Test creating a new Dataset
    dataset = H5Reference.new("/dataset1", hdf5_file_populated)
    assert isinstance(dataset, Dataset)

    # Test specifying the type
    group_specified = H5Reference.new("/group1", hdf5_file_populated, _type=Group)
    assert isinstance(group_specified, Group)

    # Test with invalid path
    with pytest.raises(KeyError):
        H5Reference.new("/invalid_path", hdf5_file_populated)


def test_h5ref_getitem_string_key(hdf5_file_populated: HDF5File):
    """Test that __getitem__ with a string key calls new() with the correct path."""
    ref = H5Reference("/", hdf5_file_populated)
    with patch.object(H5Reference, "new") as mock_new:
        ref["group1"]
        mock_new.assert_called_once_with("/group1", hdf5_file_populated, Group)


def test_h5ref_getitem_nested_path(hdf5_file_populated: HDF5File):
    """Test that __getitem__ with a nested path constructs the path correctly."""
    ref = H5Reference("/group2", hdf5_file_populated)
    with patch.object(H5Reference, "new") as mock_new:
        ref["dataset2"]
        mock_new.assert_called_once_with(
            "/group2/dataset2", hdf5_file_populated, Dataset
        )


def test_h5ref_getitem_with_explicit_type(hdf5_file_populated: HDF5File):
    """Test that __getitem__ with explicit type passes the type to new()."""
    ref = H5Reference("/", hdf5_file_populated)
    with patch.object(H5Reference, "new") as mock_new:
        ref["group1", Group]
        mock_new.assert_called_once_with("/group1", hdf5_file_populated, Group)


@pytest.mark.parametrize(
    "slice_, expected",
    [
        (slice(None), DATASET1_DATA),
        (slice(1, 2), DATASET1_DATA[1:2]),
        (slice(None, None, 2), DATASET1_DATA[::2]),
    ],
)
def test_h5ref_getitem_slice_passes_to_dataset(
    hdf5_file_populated: HDF5File, slice_: slice, expected
):
    """Test that slicing a dataset directly accesses the h5py Dataset."""
    dataset_ref = Dataset("/dataset1", hdf5_file_populated)
    with patch(
        "h5py.Dataset.__getitem__", wraps=h5py.Dataset.__getitem__, autospec=True
    ) as mock_getitem:
        res = dataset_ref[slice_]
        np.testing.assert_array_equal(res, expected)
        mock_getitem.assert_called_once()


def test_h5ref_getitem_slice_on_group_error(hdf5_file_populated: HDF5File):
    """Test that slicing a group raises an error."""
    group_ref = Group("/group1", hdf5_file_populated)
    with group_ref.open("r"):
        with pytest.raises(TypeError):
            group_ref[:]


def test_h5ref_open_delegates_to_file(hdf5_file):
    """Test that open() delegates to the file's open() method."""
    ref = H5Reference("/path/to/object", hdf5_file)
    with patch.object(hdf5_file, "open") as mock_open:
        ref.open("r", driver="mpio")
        mock_open.assert_called_once_with("r", driver="mpio")


def test_h5ref_attrs_cache(hdf5_file_populated: HDF5File):
    ref = H5Reference("/", hdf5_file_populated)
    attrs1 = ref.attrs
    attrs2 = ref.attrs
    assert attrs1 is attrs2


def test_h5ref_parent(hdf5_file_populated: HDF5File):
    ref = H5Reference("/group2/dataset2", hdf5_file_populated)
    parent = ref.parent
    assert parent._path == "/group2"
    assert isinstance(parent, Group)


# -------------------------
# Test Group and Dataset
# -------------------------
@pytest.fixture
def group(hdf5_file):
    """A fixture that creates a group in an HDF5 file and returns the group object."""
    grp = Group("/group1", hdf5_file)
    grp.require_self()
    return grp


@pytest.fixture
def dataset(hdf5_file_populated):
    """A fixture that creates a dataset in an HDF5 file and returns the dataset object."""
    return Dataset("/dataset1", hdf5_file_populated)


def test_group_setitem_immutability(hdf5_file: HDF5File):
    immutable_file = HDF5File(hdf5_file._path, mutable=False)
    group = Group("/", immutable_file)
    with pytest.raises(PermissionError):
        group["a"] = 3


def test_group_setitem_array(group: Group):
    with patch.object(
        group, "add_numerical_dataset", autospec=True
    ) as mock_add_dataset:
        group["data"] = DATASET1_DATA
        mock_add_dataset.assert_called_once()


@pytest.mark.parametrize("value", [3, 2.4, "string", True])
def test_group_setitem_single_value(group: Group, value):
    group["data"] = value
    # read and check if the attribute is in the file
    with group.open("r"):
        assert group._obj.attrs["data"] == value


@pytest.mark.parametrize("value", [3, 2.4, "string", True])
def test_group_delitem(group: Group, value):
    group["data"] = value
    del group["data"]
    with group.open("r"):
        assert "data" not in group._obj.attrs.keys()


@pytest.mark.parametrize("value", [DATASET1_DATA])
def test_group_delitem_array(group: Group, value):
    group["data"] = value
    del group["data"]
    with group.open("r"):
        assert "data" not in group._obj.keys()
        # check that it was deleted from the file map too
        assert "data" not in group.keys()


def test_group_obj_error(hdf5_file_populated: HDF5File[Mutable]):
    """test that an error is raised when trying to access group object when its not a group"""
    with pytest.raises(ValueError):
        group = Group("/dataset1", hdf5_file_populated)
        print(group)


def test_group_keys(tmp_path):
    """Test that calling keys() on a group always returns the keys."""
    filepath = tmp_path.joinpath("test.h5")
    file = HDF5File(filepath, mutable=True)
    # create a group at the root of the file
    file.root.require_group("group1")
    # dump the file object
    del file

    file_2 = HDF5File(filepath, mutable=False)
    group = file_2.root
    assert len(group.keys()) == 1


def test_group_html_repr_no_error(group):
    """Test that the HTML repr of a group does not raise an error."""
    assert group._repr_html_()


def test_group_require_self(hdf5_file):
    grp = Group("/group1", hdf5_file)
    grp.require_self()
    with grp.open("r"):
        assert "group1" in grp._file.keys()


def test_group_require_group(group):
    new_grp = group.require_group("subgroup")
    with group.open("r"):
        assert "subgroup" in group._obj.keys()
        assert "subgroup" in group.keys()
        assert isinstance(new_grp, Group)


def test_group_require_group_type_specified(group):
    class MyGroup(Group):
        pass

    new_grp = group.require_group("subgroup", return_type=MyGroup)
    assert isinstance(new_grp, MyGroup)


def test_group_require_dataset(group: Group):
    with group.open("a"):
        new_dataset = group.require_dataset("data", (2, 2), np.int64)
    assert "data" in group.datasets()


@pytest.mark.parametrize(
    "vec, dtype",
    [
        (DATASET1_DATA, np.int64),
        (DATASET2_DATA, np.float64),
    ],
)
def test_group_add_numerical_dataset(group, vec, dtype):
    """Caution: Does not test MPI implementation."""
    group.add_numerical_dataset("data", vec, attrs={"attr1": 3}, dtype=dtype)
    with group.open("r"):
        dataset = group._obj["data"]
        np.testing.assert_array_equal(dataset[:], vec)
        assert dataset.attrs["attr1"] == 3
        assert dataset.dtype == dtype


def test_dataset_shape(dataset):
    assert dataset.shape == DATASET1_DATA.shape


def test_dataset_dtype(dataset):
    assert dataset.dtype == DATASET1_DATA.dtype
