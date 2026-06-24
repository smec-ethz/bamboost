from __future__ import annotations

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Any, Protocol

from bamboost.core.simulation import Simulation
from bamboost._logger import BAMBOOST_LOGGER


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
    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name

    def resolve(self, run_state: RunState) -> Any:
        if self.attribute_name == "self":
            return run_state.context

        return getattr(run_state.context, self.attribute_name)


class FromJob(TaskInput):
    def __init__(self, job_name: str):
        self.job_name = job_name

    def resolve(self, run_state: RunState) -> Any:
        return run_state.runner._execute_job(self.job_name, run_state)


class Job:
    def __init__(
        self,
        name: str,
        func: Callable,
        inputs: dict[str, Any],
        is_persistent: bool,
        linked_sim: str | None = None,
        default: bool = False,
    ):
        self.name = name
        self.func = func
        self.input_schema = inputs
        self._is_persistent = is_persistent
        self.linked_sim = linked_sim
        self.default = default

    def execute(self, resolved_inputs: dict[str, Any]) -> Any:
        return self.func(**resolved_inputs)


class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()


class ContextProtocol(Protocol):
    sim: Simulation

@dataclass
class ProfilingStats:
    start_time: float | None = None
    end_time: float | None = None
    execution_duration: float = 0.0
    io_duration: float = 0.0
    was_cached: bool = False

@dataclass
class JobRecord:
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    exception: Exception | None = None
    stats: ProfilingStats = field(default_factory=ProfilingStats)

    @classmethod
    def from_sim_metadata(cls, job_name: str, sim: Simulation) -> JobRecord:
        record = cls()
        status_key = f"job_{job_name}_status"

        status_str = sim.metadata.get(status_key, None)
        record.status = JobStatus(status_str) if status_str else JobStatus.PENDING
        return record


class FromDictParsable(Protocol):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FromDictParsable: ...


@dataclass(frozen=True)
class Context[CFG: FromDictParsable]:
    sim: Simulation
    config: CFG


@dataclass
class RunState:
    runner: SimulationRunner
    context: Context
    records: dict[str, JobRecord]


class SimulationRunner[CFG: FromDictParsable]:
    """Orchestrates the execution of a multi-job simulation.

    Attributes:
        name: The identifier for the simulation runner.
        jobs: A collection of registered jobs.
    """

    def __init__(self, name: str, config_cls: type[CFG]):
        self.name = name
        self.jobs: dict[str, Job] = {}
        self.config_cls = config_cls

    def add_job(
        self,
        name: str | None = None,
        func: Callable | None = None,
        inputs: dict[str, Any] | None = None,
        is_persistent: bool = False,
        linked_sim: str | None = None,
        default: bool = False,
    ):
        """Registers a new job to the runner. Can be used as a method or a decorator.

        Args:
            name: The unique identifier for the job.
            func: The function to execute for this job.
            inputs: A dictionary mapping input keys to their values or providers.
        """
        actual_inputs = inputs if inputs is not None else {}

        def decorator(f: Callable) -> Callable:
            job_name = name or getattr(f, "__name__")
            self.jobs[job_name] = Job(
                job_name, f, actual_inputs, is_persistent, linked_sim, default
            )
            return f

        if func is not None and callable(func):
            return decorator(func)

        return decorator

    def run(self, sim: Simulation, targets: list[str] | None = None) -> RunState:
        """Runs the simulation pipeline.

        Args:
            context: The context one wants to access throughout the run.
            targets: List of jobs to run. Defaults to all jobs.
        """

        # Instantiate the RunState object for this specific run.
        # Initialize the records, making sure the persistent jobs states are taken from disk
        records = {job: JobRecord.from_sim_metadata(job, sim) for job in self.jobs}
        context = Context(
            sim=sim, config=self.config_cls.from_dict(sim.parameters.read())
        )
        run_state = RunState(self, context, records)

        target_jobs = targets or [key for key, job in self.jobs.items() if job.default]
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

        logger.info(
            f"Runner '{self.name}' is starting job '{job.name}' with status '{record.status}'."
        )

        # Short circuit either from cache or file
        if record.status == JobStatus.FINISHED and job._is_persistent:
            record.stats.was_cached = True
            if job.linked_sim:
                return sim.links[job.linked_sim]
            else:
                return sim
        elif record.status == JobStatus.FINISHED and not job._is_persistent:
            record.stats.was_cached = True
            return record.result

        record.stats.start_time = time.perf_counter()

        try:
            resolved_inputs = {
                k: (v.resolve(run_state) if isinstance(v, TaskInput) else v)
                for k, v in job.input_schema.items()
            }

            logger.info(f"Runner '{self.name}' is executing job '{job.name}'.")
            exec_start = time.perf_counter()
            result = job.execute(resolved_inputs)
            record.result = result
            record.stats.execution_duration = time.perf_counter() - exec_start

            # Mark the Job as 'FINISHED'
            record.status = JobStatus.FINISHED

            # Persistent jobs mutate the Simulation directly. We update H5 metadata and
            # return the Simulation instance to maintain a reference to the active state.
            if job._is_persistent:
                simw = sim.edit()
                with simw.open("a"):
                    simw.metadata[f"job_{job_name}_status"] = JobStatus.FINISHED.value

                if job.linked_sim:
                    return sim.links[job.linked_sim]
                else:
                    return sim

            record.stats.end_time = time.perf_counter()
            # Transient jobs return artifacts stored in the record for memory-based access.
            return result

        except Exception as e:
            record.status = JobStatus.FAILED
            record.exception = e
            raise e
