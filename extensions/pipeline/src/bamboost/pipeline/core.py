from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Any, Literal, Protocol

from bamboost._logger import BAMBOOST_LOGGER

if TYPE_CHECKING:
    from bamboost.core.simulation import Simulation


logger = BAMBOOST_LOGGER.getChild("pipeline-extension")


class TaskInput:
    """Base class for lazy-evaluated inputs."""

    def resolve(self, run_state: RunState) -> Any:
        raise NotImplementedError


class FromCallable(TaskInput):
    def __init__(self, func: Callable[[], Any]):
        self.func = func

    def resolve(self, run_state: RunState) -> Any:
        return self.func()


class FromContext(TaskInput):
    def __init__(self, variable_name: ContextKeys):
        self.variable_name = variable_name

    def resolve(self, run_state: RunState) -> Any:
        return getattr(run_state.context, self.variable_name)


class FromJob(TaskInput):
    def __init__(self, job_name: str):
        self.job_name = job_name

    def resolve(self, run_state: RunState) -> Any:
        return run_state.runner._execute_job(self.job_name, run_state)


class Job:
    def __init__(self, name: str, func: Callable, inputs: dict[str, Any], is_persistent: bool):
        self.name = name
        self.func = func
        self.input_schema = inputs
        self._is_persistent = is_persistent

    def execute(self, resolved_inputs: dict[str, Any]) -> Any:
        return self.func(**resolved_inputs)


class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()


class FromDictParsable(Protocol):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FromDictParsable: ...


ContextKeys = Literal["sim", "config"]

@dataclass(frozen=True)
class Context[CT: FromDictParsable]:
    sim: Simulation
    config: FromDictParsable


@dataclass
class JobRecord:
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    exception: Exception | None = None

    @classmethod
    def from_sim_metadata(cls, job_name: str, sim: Simulation) -> JobRecord:
        record = cls()
        status_key = f"stage_{job_name}_status"

        status_str = sim.metadata.get(status_key, None)
        record.status = JobStatus(status_str) if status_str else JobStatus.PENDING
        return record


@dataclass
class RunState:
    runner: SimulationRunner
    context: Context
    records: dict[str, JobRecord]


class SimulationRunner[CT: FromDictParsable]:
    """Orchestrates the execution of a multi-job simulation.

    Attributes:
        name: The identifier for the simulation runner.
        config_cls: The class used to parse and validate simulation parameters.
        jobs: A collection of registered jobs.
    """
    def __init__(self, name: str, config_cls: type[CT]):
        self.name = name
        self.config_cls = config_cls
        self.jobs: dict[str, Job] = {}

    def add_job(self, name: str, func: Callable, inputs: dict[str, Any], is_persistent: bool = False):
        """Registers a new job to the runner.

        Args:
            name: The unique identifier for the job.
            func: The function to execute for this job.
            inputs: A dictionary mapping input keys to their values or providers.
        """
        self.jobs[name] = Job(name, func, inputs, is_persistent)

    def run(self, sim: Simulation, targets: list[str] | None = None) -> RunState:
        """Runs the simulation pipeline.

        Args:
            sim: The bamboost simulation instance for the run.
            targets: List of jobs to run. Defaults to all jobs.
        """

        # Instantiate the RunState object for this specific run.
        context = Context(
            sim=sim, config=self.config_cls.from_dict(sim.parameters.read())
        )
        # Initialize the records, making sure the persistent jobs states are taken from disk
        records = {job: JobRecord.from_sim_metadata(job, sim) for job in self.jobs}
        run_state = RunState(self, context, records)

        target_jobs = targets or list(self.jobs.keys())
        for job in target_jobs:
            self._execute_job(job, run_state)

        return run_state
        
    def _execute_job(self, job_name: str, run_state: RunState) -> Any:
        """Manages the lifecycle and execution of a specific job.

        Args:
            job_name: The name of the job to execute.
            run_state: The current state object containing context and records.

        Raises:
            Exception: Propagates any error encountered during execution after recording the failure.

        Returns:
            The result of the executed job.
        """
        record = run_state.records[job_name]
        sim = run_state.context.sim
        job = self.jobs[job_name]

        logger.info(f"Runner '{self.name}' is starting job '{job.name}' with status '{record.status}'.")

        # Short circuit either from cache or file
        if record.status == JobStatus.FINISHED and job._is_persistent:
            return sim
        if record.status == JobStatus.FINISHED and not job._is_persistent:
            return record.result


        try:
            resolved_inputs = {
                k: (v.resolve(run_state) if isinstance(v, TaskInput) else v)
                for k, v in job.input_schema.items()
            }

            logger.info(f"Runner '{self.name}' is executing job '{job.name}'.")
            result = job.execute(resolved_inputs)

            # Mark the Job as 'FINISHED'
            record.status = JobStatus.FINISHED

            # Persistent jobs mutate the Simulation directly. We update H5 metadata and
            # return the Simulation instance to maintain a reference to the active state.
            if job._is_persistent:
                simw = sim.edit()
                with simw.open("a"):
                    simw.metadata[f"stage_{job_name}_status"] = JobStatus.FINISHED.value

                # Store the uid of the simulation.
                record.result = simw.uid
                return sim

            # Transient jobs return artifacts stored in the record for memory-based access.
            record.result = result
            return result

        except Exception as e:
            record.status = JobStatus.FAILED
            record.exception = e
            raise e
