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

def assert_file_open(method):
    """Decorator.
    Open/close file if memmap is False.
    Assert that file is open if memmap is True.
    """
    @wraps(method)
    def inner(self, *args, **kwargs):
        if self._file_object:
            return method(self, *args, **kwargs)
        else:
            with self.open('r') as self._file_object:
                return method(self, *args, **kwargs)
    return inner
