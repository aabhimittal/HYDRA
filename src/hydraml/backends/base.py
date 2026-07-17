"""Backend abstraction and capability negotiation.

Every backend declares a set of :class:`Capability` flags. Validation compares
what a pipeline *asks for* against what the target backend *can deliver* and
reports the difference as issues - before anything is generated or run. This
is the mechanism that keeps HYDRA honest: features never silently disappear
when you switch orchestrators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..errors import CompilationError
from ..ir.graph import structural_issues
from ..ir.model import Pipeline
from ..issues import Issue, Severity, errors


class Capability(str, Enum):
    RUNTIME_PARAMS = "runtime_params"  # params resolved at run time, not compile time
    CRON_SCHEDULE = "cron_schedule"  # schedule embedded in the compiled definition
    RETRIES = "retries"
    RETRY_BACKOFF_FACTOR = "retry_backoff_factor"  # arbitrary multiplier, not just on/off
    TASK_CACHING = "task_caching"
    PER_TASK_RESOURCES = "per_task_resources"
    GPU_RESOURCES = "gpu_resources"
    MULTI_OUTPUT = "multi_output"


@dataclass
class CompileResult:
    files: list[Path]
    issues: list[Issue] = field(default_factory=list)


class Backend(ABC):
    """A compilation (or execution) target for the HYDRA IR."""

    #: Unique registry key, e.g. "airflow".
    name: str = ""
    #: Capabilities this backend can honor natively.
    capabilities: frozenset[Capability] = frozenset()
    #: Codegen backends need task functions importable as `module:attr`.
    requires_importable_functions: bool = True

    def __init__(self, **options: Any):
        unknown = set(options) - set(self.option_names())
        if unknown:
            raise ValueError(
                f"backend {self.name!r} does not accept option(s):"
                f" {', '.join(sorted(unknown))}"
                + (f" (accepts: {', '.join(self.option_names())})" if self.option_names() else "")
            )
        self.options = options

    @classmethod
    def option_names(cls) -> tuple[str, ...]:
        return ()

    # --- validation -----------------------------------------------------------

    def validate(self, pipeline: Pipeline) -> list[Issue]:
        """Structural checks + capability gap analysis + backend extras."""
        issues = structural_issues(pipeline)
        issues.extend(self._capability_issues(pipeline))
        issues.extend(self.extra_issues(pipeline))
        return issues

    def _capability_issues(self, pipeline: Pipeline) -> list[Issue]:
        caps = self.capabilities
        issues: list[Issue] = []

        if pipeline.params and Capability.RUNTIME_PARAMS not in caps:
            issues.append(
                Issue(
                    Severity.INFO,
                    "CAP001",
                    f"{self.name}: parameters are bound at compile time; pass"
                    " --param to override, then recompile",
                )
            )
        if pipeline.schedule and Capability.CRON_SCHEDULE not in caps:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "CAP002",
                    f"{self.name}: cron schedule {pipeline.schedule!r} cannot be"
                    " embedded in the compiled definition; " + self.schedule_hint(),
                )
            )

        for tname, task in pipeline.tasks.items():
            if task.retry.retries > 0 and Capability.RETRIES not in caps:
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "CAP003",
                        f"{self.name}: retries are not supported and will be dropped",
                        task=tname,
                    )
                )
            if (
                task.retry.retries > 0
                and task.retry.backoff != 1.0
                and Capability.RETRY_BACKOFF_FACTOR not in caps
                and Capability.RETRIES in caps
            ):
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "CAP004",
                        f"{self.name}: backoff factor {task.retry.backoff} is"
                        " approximated (backend supports only on/off exponential"
                        " backoff)",
                        task=tname,
                    )
                )
            if task.cache and Capability.TASK_CACHING not in caps:
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "CAP005",
                        f"{self.name}: task caching is not supported; the task"
                        " will re-run on every pipeline run",
                        task=tname,
                    )
                )
            if not task.resources.is_empty and Capability.PER_TASK_RESOURCES not in caps:
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "CAP006",
                        f"{self.name}: per-task resource requests are not"
                        " enforced by this backend",
                        task=tname,
                    )
                )
            if task.resources.gpu > 0 and Capability.GPU_RESOURCES not in caps:
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "CAP007",
                        f"{self.name}: GPU request ({task.resources.gpu}) cannot"
                        " be honored by this backend",
                        task=tname,
                    )
                )
            if not task.single_output and Capability.MULTI_OUTPUT not in caps:
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "CAP008",
                        f"{self.name}: multi-output tasks are not supported",
                        task=tname,
                    )
                )
            if self.requires_importable_functions and not _importable(task.fn_ref):
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "CAP009",
                        f"{self.name}: task function {task.fn_ref!r} is not"
                        " importable as `from module import name` (nested or"
                        " locally-defined functions cannot be compiled)",
                        task=tname,
                    )
                )
        return issues

    def extra_issues(self, pipeline: Pipeline) -> list[Issue]:
        """Backend-specific validation beyond the shared capability checks."""
        return []

    def schedule_hint(self) -> str:
        return "configure the schedule in the target system"

    # --- compilation ----------------------------------------------------------

    def compile(
        self,
        pipeline: Pipeline,
        outdir: Path,
        params: dict[str, Any] | None = None,
    ) -> CompileResult:
        issues = self.validate(pipeline)
        if errors(issues):
            raise CompilationError(
                f"cannot compile pipeline {pipeline.name!r} for backend"
                f" {self.name!r}: validation failed",
                issues=issues,
            )
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        files = self.emit(pipeline, outdir, pipeline.resolve_params(params))
        return CompileResult(files=files, issues=issues)

    @abstractmethod
    def emit(self, pipeline: Pipeline, outdir: Path, params: dict[str, Any]) -> list[Path]:
        """Write compiled artifact(s) and return their paths."""


def _importable(fn_ref: str) -> bool:
    module, _, qualname = fn_ref.partition(":")
    return bool(module) and module != "__main__" and "." not in qualname and "<" not in qualname


# --- registry ------------------------------------------------------------------

_REGISTRY: dict[str, type[Backend]] = {}


def register(cls: type[Backend]) -> type[Backend]:
    if not cls.name:
        raise ValueError("backend class must set a name")
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str, **options: Any) -> Backend:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r} (available: {', '.join(sorted(_REGISTRY))})"
        ) from None
    return cls(**options)


def backend_names() -> list[str]:
    return sorted(_REGISTRY)
