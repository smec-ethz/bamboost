# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2024 Flavio Lorez and contributors
#
# There is no warranty for this code

from __future__ import annotations

import os
import subprocess
from functools import wraps

from bamboost.simulation_writer import SimulationWriter


def _extend_exit_slurm_info(original_exit):
    @wraps(original_exit)
    def modified_exit(self: SimulationWriter, exc_type, exc_value, exc_tb):
        # Inject the following into the __exit__ method of SimulationWriter
        slurm_job_id = os.environ.get("SLURM_JOB_ID")

        result = subprocess.run(
            ["myjobs", "-j", slurm_job_id], env=os.environ, capture_output=True
        )

        # Check if the command was successful
        if result.returncode == 0:
            # Get the output as a string
            output_str = result.stdout.decode("utf-8")

            # Create a dictionary to store the key-value pairs
            slurm_dict = {}
            for line in output_str.strip().split("\n"):
                if ":" in line:  # Ensure the line contains a key-value pair
                    key, value = line.split(":", 1)  # Split on the first colon
                    slurm_dict[key.strip()] = value.strip()

            _write_slurm_info(self, slurm_dict)

        return original_exit(self, exc_type, exc_value, exc_tb)

    return modified_exit


def _write_slurm_info(self: SimulationWriter, slurm_dict: dict):
    """Write the slurm metadata to the HDF5 file.

    Args:
        - self: The SimulationWriter instance.
        - slurm_dict: A dictionary containing the key-value pairs extracted
          from the SLURM job.
    """
    # Create a group for the SLURM metadata
    if self._prank == 0:
        with self.open("a") as file:
            slurm_group = file.file_object.require_group("slurm")
            slurm_group.attrs.update(slurm_dict)


def install():
    """Install the slurm extension to the SimulationWriter class. Extends the
    __exit__ method to add slurm metadata.
    """
    SimulationWriter.__exit__ = _extend_exit_slurm_info(SimulationWriter.__exit__)

