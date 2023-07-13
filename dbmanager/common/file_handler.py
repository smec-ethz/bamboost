import h5py
import time


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

            try:
                if driver=='mpio' and h5py._MPI_ACTIVE:
                    return h5py.File(file, mode, driver=driver, comm=comm)
                else:
                    return h5py.File(file, mode)
            except AttributeError:  # If h5py._MPI_ACTIVE is unavailable
                return h5py.File(file, mode)

        except OSError:
            time.sleep(1)
            print(f'File {file} not accessible, waiting...', end='\r') 

