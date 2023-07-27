import time
import h5py
import logging

log = logging.getLogger(__name__)

HAS_MPIO = 'mpio' in h5py.registered_drivers()


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
            if driver=='mpio' and HAS_MPIO:
                return h5py.File(file, mode, driver=driver, comm=comm)
            else:
                return h5py.File(file, mode)

        except OSError:
            log.info(f"File {file} not accessible, waiting") 
            time.sleep(1)
