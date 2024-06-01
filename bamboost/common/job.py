# This file is part of bamboost, a Python library built for datamanagement
# using the HDF5 file format.
#
# https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager
#
# Copyright 2023 Flavio Lorez and contributors
#
# There is no warranty for this code

import argparse
import os

from bamboost._config import paths

__all__ = ["Job"]


class Job:
    def __init__(self):
        pass

    def _set_environment_variables(self, db_id: str, uid: str) -> str:
        # access sqlite database and get path
        tmp_str = ""
        tmp_str += f"""DATABASE_DIR=$(sqlite3 {paths['DATABASE_FILE']} "SELECT path FROM dbindex WHERE id='{db_id}'")\n"""

        # set further environment variables
        tmp_str += f"SIMULATION_DIR=$DATABASE_DIR/{uid}\n"
        tmp_str += f"SIMULATION_ID={db_id}:{uid}\n\n"
        return tmp_str

    def create_sbatch_script(
        self,
        commands: list,
        path: str,
        uid: str = None,
        db_id: str = None,
        nnodes: int = 1,
        ntasks: int = 4,
        ncpus: int = 1,
        time: str = "04:00:00",
        mem_per_cpu: int = 2048,
        tmp: int = 8000,
    ) -> None:
        """Write sbatch script for new simulation."""
        nb_tasks_per_node = int(ntasks / nnodes)

        # define how mpirun is called
        mpicommand = ""
        if ntasks > 1:
            mpicommand = "mpirun "

        script = f"#!/bin/bash\n\n"

        # sbatch commands
        script += f"#SBATCH --ntasks={ntasks}\n"
        if ntasks > 1:
            script += f"#SBATCH --nodes={nnodes}\n"
            script += f"#SBATCH --cpus-per-task={ncpus}\n"
            script += f"#SBATCH --ntasks-per-node={nb_tasks_per_node}\n"
        script += f"#SBATCH --time={time}\n"
        script += f"#SBATCH --job-name={uid}\n"
        script += f"#SBATCH --mem-per-cpu={mem_per_cpu}\n"
        if tmp:
            script += f"#SBATCH --tmp={tmp}\n"
        script += f"#SBATCH --output={os.path.join(path, uid)}/{uid}.out\n"

        # add SCRIPT_DIR as environment variable
        script += f"""path=$(sqlite3 {paths['DATABASE_FILE']} "SELECT path FROM dbindex WHERE id='{db_id}'")\n"""
        script += f"SCRIPT_DIR=$path/{uid}\n\n"

        # user defined commands
        script += "\n"
        for cmd in commands:
            script += cmd.format(MPI=mpicommand) + "\n"

        # write to submission file
        with open(
            os.path.join(os.path.join(path, uid), f"sbatch_{uid}.sh"), "w"
        ) as file:
            file.write(script)

    def create_bash_script_local(
        self, commands: list, path: str, uid: str, db_id: str = None, ntasks: int = 4
    ) -> None:
        """Write bash script for local execution.

        The following environment variables are written to the script:
            - DATABASE_DIR: Path to the database directory.
            - SIMULATION_DIR: Path to the simulation directory.
            - SIMULATION_ID: Full id of the simulation.

        Args:
            commands (list): List of commands to be executed.
            path (str): Path to the database directory.
            uid (str): Unique identifier of the simulation.
            db_id (str): Database ID. Defaults to None.
            ntasks (int): Number of tasks. Defaults to 4.
        """

        # define how mpirun is called
        mpicommand = ""
        if ntasks > 1:
            mpicommand = f"mpirun -n {ntasks}"

        script = f"#!/bin/bash\n\n"

        # add paths as environment variables
        script += self._set_environment_variables(db_id, uid)

        # user defined commands
        for cmd in commands:
            script += cmd.format(MPI=mpicommand) + "\n"

        with open(os.path.join(os.path.join(path, uid), f"{uid}.sh"), "w") as file:
            file.write(script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_to_run", type=str)
    # parser.add_argument('input_file', type=str)
    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks", type=int, default=4)
    parser.add_argument("--ncpus", type=int, default=1)
    parser.add_argument("--time", type=str, default="04:00:00")
    parser.add_argument("--mem_per_cpu", type=int, default=2048)
    args = parser.parse_args()

    job = Job()
    job.create_sbatch_script(
        [args.file_to_run],
        args.uid,
        args.nodes,
        args.ntasks,
        args.ncpus,
        args.time,
        args.mem_per_cpu,
    )
