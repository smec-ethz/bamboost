# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2024 Flavio Lorez and contributors
#
# There is no warranty for this code


class MockMPI:
    """
    Mock class for `mpi4py.MPI` to be used when MPI is not available or usage
    not desired. Not importing MPI increases launch speed significantly, which
    is important for CLI applications.

    Attributes:
        Comm: Mock class for `mpi4py.MPI.Comm`
        COMM_WORLD: Mock object for `mpi4py.MPI.COMM_WORLD`
        COMM_SELF: Mock object for `mpi4py.MPI.COMM_SELF`
        COMM_NULL: Mock object for `mpi4py.MPI.COMM_NULL`
    """

    class Comm:
        def __init__(self):
            self.size = 1
            self.rank = 0
            self.comm = None
            self.is_mpi = False
            self.is_master = True

        def barrier(self):
            pass

        def finalize(self):
            pass

        def bcast(self, data, root=0):
            return data

        def scatter(self, data, root=0):
            return data

        def gather(self, data, root=0):
            return data

        def allgather(self, data):
            return [data]

        def allreduce(self, data, op):
            return data

        def reduce(self, data, op, root=0):
            return data

        def send(self, data, dest, tag=0):
            pass

        def recv(self, source, tag=0):
            return None

        def recv_any_source(self, tag=0):
            return None

        def sendrecv(self, send_data, dest, tag=0):
            return None

        def sendrecv_replace(self, data, dest, tag=0):
            return data

        def get_processor_name(self):
            return "localhost"

        def get_version(self):
            return "0.0.0"

        def get_library_version(self):
            return "0.0.0"

        def get_error_string(self, errorcode):
            return "No error"

        def get_exception_class(self):
            return Exception

        def get_exception_string(self, errorcode):
            return "No error"

        def get_count(self, status, datatype):
            return 0

        def get_status(self, request):
            return None

        def get_source(self, status):
            return 0

        def get_tag(self, status):
            return 0

        def get_elements(self, status):
            return 0

        def get_bytes(self, status):
            return 0

        def get_cancelled(self, status):
            return False

        def get_topo(self):
            return None

        def get_cart(self):
            return None

        def get_dims(self):
            return None

        def get_coords(self):
            return None

        def get_rank(self):
            return 0

        def get_size(self):
            return 1

        def Get_size(self):
            return 1

    COMM_WORLD = Comm()
    COMM_SELF = Comm()
    COMM_NULL = Comm()
