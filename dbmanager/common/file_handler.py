from itertools import cycle
import time
import sys
import h5py


def open_h5file(file: str, mode, driver=None, comm=None):
    """Open h5 file. Waiting if file is not available.

    Args:
        file (str): File to open
        mode (str): 'r', 'a', 'w', ...
        driver (str): driver for h5.File
        comm: MPI communicator
    """
    dots = cycle(['.','..','...'])
    while True:
        try:
            if driver=='mpio' and ('mpio' in h5py.registered_drivers()):
                return h5py.File(file, mode, driver=driver, comm=comm)
            else:
                return h5py.File(file, mode)

        except OSError:
            print(f"File {file} not accessible, waiting{next(dots)}", flush=True, end="\r") 
            time.sleep(1)
            sys.stdout.write("\033[K")
