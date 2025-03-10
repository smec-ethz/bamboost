from pathlib import Path, PurePosixPath
from unittest.mock import patch

import h5py
import pytest

from bamboost._typing import Mutable
from bamboost.core.hdf5 import HDF5File, HDF5Path
from bamboost.core.hdf5.filemap import FileMap, FilteredFileMap


@pytest.fixture
def hdf5_file(tmp_path: Path):
    return HDF5File(tmp_path.joinpath("test.h5"), mutable=True)


@pytest.fixture(scope="module")
def hdf5_file_module(tmp_path_module: Path):
    return HDF5File(tmp_path_module.joinpath("test.h5"), mutable=True)


# ------------------------------
# HDF5Path
# ------------------------------
def test_hdf5_path():
    assert HDF5Path("path/to/file", absolute=True) == "/path/to/file"
    assert HDF5Path("//////path////to//file", absolute=True) == "/path/to/file"
    assert HDF5Path("path/to/file", absolute=False) == "path/to/file"
    assert HDF5Path("/path/to/file", absolute=False) == "path/to/file"


def test_hdf5_path_relative_to():
    path = HDF5Path("/path/to/file")
    assert path.relative_to("path") == "to/file"
    assert path.relative_to("path/to") == "file"


def test_hdf5_path_relative_to_error():
    path = HDF5Path("/path/to/file")
    with pytest.raises(ValueError):
        path.relative_to("not_a_parent")


def test_hdf5_path_join():
    path = HDF5Path("/path/to/file")
    assert path.joinpath("another") == "/path/to/file/another"
    # also testing the __truediv__ operator
    assert path / "another" == "/path/to/file/another"


def test_hdf5_path_parent():
    path = HDF5Path("/path/to/file")
    assert path.parent == "/path/to"


def test_hdf5_path_basename():
    path = HDF5Path("/path/to/file")
    assert path.basename == "file"


def test_hdf5_path_posix():
    path = HDF5Path("/path/to/file")
    assert isinstance(path.path, PurePosixPath)
    assert path.path.stem == "file"


# ------------------------------
# FileMap
# ------------------------------
@pytest.fixture(scope="module")
def filemap(hdf5_file_module: HDF5File):
    # add some groups and datasets using h5py directly
    # lenght: 5
    with hdf5_file_module.open("w") as f:
        f.create_group("group1")
        grp2 = f.create_group("group2")
        f.create_dataset("dataset1", data=[1, 2, 3])
        grp2.create_dataset("dataset2", data=[4, 5, 6])
        grp2.create_group("group3")

    # check that the file map is populated
    fm = FileMap(hdf5_file_module)
    with hdf5_file_module.open("r"):
        fm.populate()
    return fm


def test_filemap_populate(hdf5_file: HDF5File):
    # add some groups and datasets using h5py directly
    with hdf5_file.open("w") as f:
        f.create_group("group1")
        f.create_dataset("dataset1", data=[1, 2, 3])

    # check that the file map is populated
    fm = FileMap(hdf5_file)
    with hdf5_file.open("r"):
        with patch("h5py.File.visititems", wraps=fm._file.visititems) as visititems:
            fm.populate()

    # check that the visititems method was called
    visititems.assert_called_once()
    # assert dict is not empty
    assert fm._dict


def test_filemap_populate_error(hdf5_file: HDF5File):
    # check that the method raises an error if the file is not open or was never opened
    with pytest.raises((ValueError, AttributeError)):
        FileMap(hdf5_file).populate()


def test_filemap_getitem(filemap: FileMap):
    # check the entries
    assert filemap["/group1"] == h5py.Group
    assert filemap["/group2"] == h5py.Group
    assert filemap["/group2/dataset2"] == h5py.Dataset
    assert filemap["dataset1"] == h5py.Dataset


def test_filemap_len(filemap: FileMap):
    assert len(filemap) == 5


def test_filemap_iter(filemap: FileMap):
    for key in filemap:
        assert key in [
            "/group1",
            "/group2",
            "/group2/dataset2",
            "/group2/group3",
            "/dataset1",
        ]


def test_filemap_datasets(filemap: FileMap):
    datasets = filemap.datasets()
    assert len(datasets) == 2
    assert "/dataset1" in datasets
    assert "/group2/dataset2" in datasets


def test_filemap_groups(filemap: FileMap):
    groups = filemap.groups()
    assert len(groups) == 3
    assert "/group1" in groups
    assert "/group2" in groups
    assert "/group2/group3" in groups


def test_filtered_filemap(filemap: FileMap):
    filtered = FilteredFileMap(filemap, parent="group2")

    assert len(filtered) == 2
    assert filtered.datasets() == ("dataset2",)
    assert filtered.groups() == ("group3",)
    assert filtered["dataset2"] == h5py.Dataset
    assert filtered["group3"] == h5py.Group


# ------------------------------
# HDF5File
# ------------------------------
@pytest.mark.parametrize(
    "name, mutable",
    [
        ("test.h5", True),
        ("....++.h5", True),  # aweful names are not forbidden
    ],
)
def test_hdf5_file_init(tmp_path: Path, name: str, mutable: bool):
    f = HDF5File(tmp_path.joinpath(name), mutable=mutable)

    # check the attributes
    assert f._filename == tmp_path.joinpath(name).as_posix()
    assert f._path == tmp_path.joinpath(name).absolute()
    assert isinstance(f.file_map, FileMap)
    assert f.mutable == mutable


def test_hdf5_file_init_error(tmp_path: Path):
    """Test that we immediately raise an exception if the file is immutable and does not exist."""
    with pytest.raises(FileNotFoundError):
        HDF5File(tmp_path.joinpath("not_existing.h5"), mutable=False)


def test_hdf5_file_open(hdf5_file: HDF5File):
    with hdf5_file.open("w") as f:
        assert f.file.mode == "r+"
        assert f.file.id.valid


def test_hdf5_file_mutability(hdf5_file: HDF5File[Mutable]):
    hdf5_file._create_file()

    # create an immutable file object and try to open it
    f_immutable = HDF5File(hdf5_file._path, mutable=False)

    with pytest.raises(PermissionError):
        f_immutable.open("w")


def test_hdf5_file_context(hdf5_file: HDF5File):
    with patch("h5py.File.__init__", wraps=h5py.File.__init__) as h5py_init:
        with hdf5_file.open("w"):
            with hdf5_file.open("w"):
                with hdf5_file.open("w"):
                    assert hdf5_file.is_open
    # check that the file was opened only once
    h5py_init.assert_called_once()


def test_hdf5_file_context_mode_change(hdf5_file: HDF5File):
    with hdf5_file.open("w") as f:
        pass  # create file
    with (
        patch("h5py.File.__init__", wraps=h5py.File.__init__) as h5py_init,
        patch("h5py.File.close", wraps=super(HDF5File, hdf5_file).close) as h5py_close,
    ):
        with hdf5_file.open("r"):
            assert hdf5_file.mode == "r"
            with hdf5_file.open("w"):
                assert hdf5_file.mode == "r+"
                with hdf5_file.open("r"):
                    assert hdf5_file.mode == "r+"  # still writable

    # check that the file was opened twice
    assert h5py_init.call_count == 2
    assert h5py_close.call_count == 2


def test_hdf5_file_close_depending_on_context(hdf5_file: HDF5File):
    hdf5_file.open("w")
    hdf5_file.open("w")
    hdf5_file.open("w")
    hdf5_file.close()
    # file should be still open with context stack 2
    assert hdf5_file.is_open
    assert hdf5_file._context_stack == 2

    # close 2 times more to close
    hdf5_file.close()
    hdf5_file.close()
    assert not hdf5_file.is_open
    assert hdf5_file._context_stack == 0

    # check that further calls to close do nothing
    hdf5_file.close()


def test_hdf5_file_force_close(hdf5_file: HDF5File):
    hdf5_file.open("w")
    hdf5_file.open("w")
    hdf5_file.open("w")
    hdf5_file.force_close()
    assert not hdf5_file.is_open
