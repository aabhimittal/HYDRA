"""Exception hierarchy for HYDRA."""

from __future__ import annotations


class HydraError(Exception):
    """Base class for all HYDRA errors."""


class UsageError(HydraError):
    """The authoring DSL was used in a way that cannot be traced into the IR."""


class CycleError(HydraError):
    """The task graph contains a cycle."""

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        super().__init__("cycle detected in task graph: " + " -> ".join([*cycle, cycle[0]]))


class LoadError(HydraError):
    """A pipeline spec string could not be resolved to a Pipeline object."""


class CompilationError(HydraError):
    """Compilation aborted because validation produced errors."""

    def __init__(self, message: str, issues=None):
        self.issues = list(issues or [])
        super().__init__(message)


class ExecutionError(HydraError):
    """A task failed during a local run after exhausting its retry budget."""

    def __init__(self, task: str, cause: BaseException):
        self.task = task
        self.cause = cause
        super().__init__(f"task {task!r} failed: {cause!r}")
