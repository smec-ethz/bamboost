import argparse
import importlib.util
import os
import sys
from types import ModuleType
from typing import Iterable


class BamboostInputFile:
    REQUIRED_ATTRIBUTES = [
        "parameters",
        "commands",
    ]
    OPTIONAL_ATTRIBUTES = [
        "files",
        "links",
        "sbatch_kwargs",
    ]

    parameters: dict
    commands: list[str]
    sbatch_kwargs: dict
    files: Iterable[str]
    links: dict[str, str]

    def __init__(self, mod: ModuleType) -> None:
        # Initialize default optional attributes
        self.files = []
        self.links = {}
        self.sbatch_kwargs = {}

        for attr in self.REQUIRED_ATTRIBUTES:
            if not hasattr(mod, attr):
                raise ValueError(f"Missing required input: {attr}")
            setattr(self, attr, getattr(mod, attr))

        for attr in self.OPTIONAL_ATTRIBUTES:
            if hasattr(mod, attr):
                setattr(self, attr, getattr(mod, attr))


def job():
    parser = argparse.ArgumentParser(description="Create and submit jobs")
    parser.add_argument("db_path", type=str, help="Path to the database")
    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("--name", "-n", type=str, help="Name of the job")
    parser.add_argument("--prefix", "-np", type=str, help="Prefix for the job name")
    parser.add_argument("--note", "-m", type=str, help="Note for the job")
    parser.add_argument(
        "--submit", "-s", action="store_true", help="Directly submit the job"
    )
    parser.add_argument("--euler", "-e", action="store_true", help="Job for Euler")

    args = parser.parse_args()

    sim = create_job(
        args.db_path,
        args.input_file,
        args.name,
        args.note,
        args.euler,
        prefix=args.prefix,
    )

    if args.submit:
        sim.submit()
        print(f"Submitted simulation {sim.uid} [db: {sim.database_id}]", flush=True)


def create_job(
    db_path: str,
    input_file: str,
    name: str = None,
    note: str = None,
    euler: bool = False,
    prefix: str = None,
):
    from bamboost import config

    config.options.mpi = False
    from bamboost.core.collection import Collection

    bb_input = _import_input_file(input_file)

    db = Collection(db_path)

    # create simulation
    sim = db.create_simulation(
        name,
        bb_input.parameters,
        skip_duplicate_check=True if name else False,
        note=note,
        files=bb_input.files,
        links=bb_input.links,
        prefix=prefix,
    )

    # create submission script
    sim.create_run_script(
        bb_input.commands,
        euler=euler,
        sbatch_kwargs=bb_input.sbatch_kwargs,
    )

    return sim


def _import_input_file(input_file: str):
    # Add the directory of the input file to sys.path
    input_dir = os.path.dirname(os.path.abspath(input_file))
    if input_dir not in sys.path:
        sys.path.insert(0, input_dir)

    module_name = "bb_input"
    spec = importlib.util.spec_from_file_location(module_name, input_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return BamboostInputFile(module)
