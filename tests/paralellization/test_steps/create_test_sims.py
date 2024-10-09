import sys

from bamboost import Manager


def create_test_run(
    db: Manager, nb_processes: int = 1, array_size: int = 10000, nb_steps: int = 100
) -> None:
    """Create a test run for given number of processes and array sizes."""
    params = {
        "nb_processes": nb_processes,
        "array_size": (array_size, 3),
        "nb_steps": nb_steps,
    }
    script_file = "script.py"
    sim = db.create_simulation(
        f"{array_size}_{nb_processes:02d}", parameters=params, skip_duplicate_check=True
    )
    mpicommand = "" if nb_processes == 1 else f"mpirun -n {nb_processes}"

    commands = [
        f"{mpicommand} python3 $SIMULATION_DIR/{script_file} --path $SIMULATION_DIR/.. --uid {sim.uid}"
    ]
    sim.create_batch_script(commands, ntasks=nb_processes, euler=False)
    sim.copy_file(script_file)


def main(test_manager_name: str = "out"):
    manager = Manager(test_manager_name)

    for nb_processes in [1, 2, 4, 8]:
        array_size = 200000
        nb_steps = 100
        create_test_run(manager, nb_processes, array_size, nb_steps)


if __name__ == "__main__":
    out_dir = sys.argv[1]
    main(out_dir)
