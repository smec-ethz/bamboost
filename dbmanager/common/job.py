import argparse
import os

class Job:

    def __init__(self):
        pass

    def create_sbatch_script(self, file_to_run: str, path: str, uid: str = None, nnodes: int = 1,
                             ntasks: int = 4, ncpus: int = 1, time: str = '04:00:00',
                             mem_per_cpu: int = 2048, tmp: int = 8000) -> None:
        """Write sbatch script for new simulation."""
        nb_tasks_per_node = int(ntasks/nnodes)

        script = f"#!/bin/bash\n\n"

        script += f"#SBATCH --ntasks={ntasks}\n"
        script += f"#SBATCH --nodes={nnodes}\n"
        script += f"#SBATCH --cpus-per-task={ncpus}\n"
        script += f"#SBATCH --ntasks-per-node={nb_tasks_per_node}\n"
        script += f"#SBATCH --time={time}\n"
        script += f"#SBATCH --job-name={uid}\n"
        script += f"#SBATCH --mem-per-cpu={mem_per_cpu}\n"
        if tmp:
            script += f"#SBATCH --tmp={tmp}\n"
        script += f"#SBATCH --output={os.path.join(path, uid)}/{uid}.out\n"

        script += '\n'
        script += f'mpirun python {file_to_run} --path {path} --uid {uid}\n'
        
        with open(os.path.join(os.path.join(path, uid), f'sbatch_{uid}.sh'), 'w') as file:
            file.write(script)

    def create_bash_script_local(self, file_to_execute: str, path: str, uid: str, ntasks: int = 4):
        """Write bash script for local execution."""

        script = f"#!/bin/bash\n\n"
        script += f"mpirun -n {ntasks} python3 {file_to_execute} --path {path} --uid {uid}\n"
        with open(os.path.join(os.path.join(path, uid), f'{uid}.sh'), 'w') as file:
            file.write(script)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file_to_run', type=str)
    # parser.add_argument('input_file', type=str)
    parser.add_argument('--uid', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--ntasks', type=int, default=4)
    parser.add_argument('--ncpus', type=int, default=1)
    parser.add_argument('--time', type=str, default='04:00:00')
    parser.add_argument('--mem_per_cpu', type=int, default=2048)
    args = parser.parse_args()

    job = Job()
    job.create_sbatch_script(args.file_to_run, args.uid, args.nodes,
                             args.ntasks, args.ncpus, args.time, args.mem_per_cpu)
