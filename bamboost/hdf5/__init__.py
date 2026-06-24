import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    [],
    {
        "file": ["FileMode", "HDF5File", "HDF5Path"],
        "filemap": ["FileMap", "FilteredFileMap"],
    },
)
