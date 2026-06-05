import logging

from bamboost._logger import BAMBOOST_LOGGER
from bamboost.cli.app import app as app
from bamboost.cli.run import Script as Script


# For the cli, any logging should be printed to the console instead.
# This is done by removing the default stream handler and adding a custom one which only
# and immmediately prints the message to the console.
class CliHandler(logging.Handler):
    def emit(self, record):
        from bamboost.mpi import MPI

        if MPI.COMM_WORLD.rank == 0:
            print(record.getMessage())


# BAMBOOST_LOGGER.addHandler(CliHandler())
BAMBOOST_LOGGER.setLevel("ERROR")
