---
title: Job management
---

Bamboost includes some tools to manage your jobs.
It can:

- Create a submission script. Either **bash** for local execution or
  **sbatch/slurm** for remote execution.
- Submit jobs on the cluster (or locally).
- Monitor the status of your jobs.
- Monitor slurm job status on _Euler_.

I suggest the following workflow:

1. Create a _job runtime_ script. This script will be the actual execution of
   your simulation. So it is all the code you do given an _input file_ (just
   that you won't have an input file).
2. Create a _job creation_ script. In this script, you define the job
   parameters and the commands to execute. Also include and copy the _job
   runtime_ script to the bamboost simulation.
3. Run the _job creation_ script. This will create a submission script that
   that can be executed or submitted on the cluster.

## Job runtime script

This is your actual simulation code.

```python title="Example script.py"
import argparse
from bamboost import SimulationWriter

def main(sim: SimulationWriter):

    parameters = sim.parameters

    sim.add_metadata()
    sim.register_git_attributes("path/to/repo")
    sim.add_mesh(...)

    # Model calculations
    for i in range(parameters["n_steps"]):
        sim.add_field(...)
        sim.add_field(...)
        sim.add_global_field(...)
        sim.finish_step()

    sim.create_xdmf_file()
    sim.finish_sim()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=str, required=True)
    args = parser.parse_args()

    with SimulationWriter.fromUID(args.simulation) as sim:
        main(sim)
```

## Job creation script

This script will create a submission script that can be executed or submitted.
Also it serves as an input file.
It should do the following:

- define parameters
- copy the _job runtime_ script to the simulation
- copy itself to the simulation
- define the commands to execute
- create the submission script
- (optional) submit the job immediately

```python title="Example job.py"
import argparse
import os
os.environ["BAMBOOST_MPI"] = "0"
from bamboost import Manager

def main(db_path, name: str = None, *, submit: bool = False):

    script_file = "path/to/runtime_script"
    ntasks = 8
    mem_per_cpu = "2G"
    note = "This is a test job"
    
    params = {
        "n_steps": 100,
        "E": 1.0,
        "nu": 0.3,
        "mesh": "<mesh-db>:<mesh-id>",
    }

    db = Manager(db_path)
    sim = db.create_simulation(
        name=name,
        parameters=params,
        skip_duplicate_check=True,
    )
    sim.change_note(note)
    sim.links["mesh"] = params["mesh"]
    sim.copy_file(script_file)
    sim.copy_file(__file__)

    cmd = [
        "source ~/envs/my-euler-modules.sh",
        f"srun python3 $SIMULATION_DIR/{os.path.basename(script_file)} --simulation {sim.get_full_uid()}",
    ]
    sim.create_run_script(
        cmd,
        ntasks=ntasks,
        mem_per_cpu=mem_per_cpu,
        euler=True,
        time="04:00:00",
    )

    if submit:
        sim.submit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=str, help="Path to the database.")
    parser.add_argument(
        "name", type=str, help="Name of the simulation.", default=None, nargs="?"
    )
    parser.add_argument("--submit", "-s", action="store_true", help="Submit the job.")
    args = parser.parse_args()

    main(args.db_path, args.name, submit=args.submit)
```
