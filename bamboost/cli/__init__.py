import logging

from bamboost._logger import BAMBOOST_LOGGER, STREAM_HANDLER
from bamboost.cli.app import app as app
from bamboost.cli.run import configure as configure
from bamboost.cli.run import execute as execute


# For the cli, any logging should be printed to the console instead.
# This is done by removing the default stream handler and adding a custom one which only
# and immmediately prints the message to the console.
class CliHandler(logging.Handler):
    def emit(self, record):
        print(record.getMessage())


BAMBOOST_LOGGER.addHandler(CliHandler())
BAMBOOST_LOGGER.removeHandler(STREAM_HANDLER)
BAMBOOST_LOGGER.setLevel("CRITICAL")
