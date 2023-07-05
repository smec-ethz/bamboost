import h5py
import time
from functools import wraps
import mpi4py


def open_h5file(file: str, mode, driver=None, comm=None):
    """Open h5 file. Waiting if file is not available.

    Args:
        file (str): File to open
        mode (str): 'r', 'a', 'w', ...
        driver (str): driver for h5.File
        comm: MPI communicator
    """
    while True:
        try:
            if driver=='mpio' and h5py._MPI_ACTIVE:
                return h5py.File(file, mode, driver=driver, comm=comm)
            else:
                return h5py.File(file, mode)
        except OSError:
            time.sleep(1)
            print(f'File {file} not accessible, waiting...', end='\r') 

def memmap_decorator(method):
    """Decorator.
    Open/close file if memmap is False.
    Assert that file is open if memmap is True.
    """
    @wraps(method)
    def inner(self, *args, **kwargs):
        if self.memmap:
            assert self._file_object, "File is not open and `memmap` is set True"
            return method(self, *args, **kwargs)
        else:
            with self.open('r') as self._file_object:
                res = method(self, *args, **kwargs)
                # If attribute is hdf5 dataset, load it into RAM
                if isinstance(res, h5py._hl.dataset.Dataset):
                    return res[()]
                if isinstance(res, tuple):
                    return (i[()] if isinstance(i, h5py._hl.dataset.Dataset) else i for i in res)
                for attr in res.__dir__():
                    if isinstance(res.__getattribute__(attr), h5py._hl.dataset.Dataset):
                        res.__setattr__(attr, res.__getattribute__(attr)[()])
                return res
    return inner
