import os

os.environ["BAMBOOST_MPI"] = "0"
from .job import job
from .main import *
