"""The HYDRA intermediate representation (IR).

The IR is the contract between the authoring DSL and the backend compilers.
It is deliberately small, backend-neutral, and JSON/YAML-serializable:

- a ``Pipeline`` is a named DAG of ``TaskSpec`` nodes plus typed ``ParamSpec``
  inputs and an optional cron schedule;
- task inputs are references (``ParamRef``, ``OutputRef``) or JSON literals
  (``LiteralValue``) - never live Python objects;
- task bodies are referenced by import path (``module:qualname``), never
  embedded, so the same IR can be compiled for any backend.

Design tenet: values that flow between tasks are *small* - parameters and
URIs. Heavy artifacts (datasets, model weights) live in external storage and
are passed by reference. This is the only data-passing model that maps cleanly
onto XCom (Airflow), task return values (Prefect), and component parameters
(Kubeflow) at the same time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

#: Closed set of IR-level types. The DSL maps Python annotations into this set;
#: backends map it onto their own type systems (or warn when they cannot).
TYPE_NAMES = ("str", "int", "float", "bool", "dict", "list", "Any")

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")


class _Required:
    """Sentinel for params without a default (required at run/compile time)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<required>"


REQUIRED = _Required()


@dataclass(frozen=True)
class Resources:
    """Per-task compute requirements.

    ``cpu`` and ``memory`` use Kubernetes quantity strings ("2", "500m",
    "4Gi") because that is the least lossy common denominator; backends that
    have no per-task resource model surface a validation issue instead of
    silently dropping these.
    """

    cpu: str | None = None
    memory: str | None = None
    gpu: int = 0
    gpu_type: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.cpu is None and self.memory is None and self.gpu == 0


@dataclass(frozen=True)
class RetryPolicy:
    """Normalized retry semantics.

    ``max_attempts`` counts *total* attempts (1 = no retries). ``backoff`` is
    a per-retry delay multiplier (1.0 = constant delay). Backends translate
    into their own vocabulary - e.g. Airflow's ``retries`` is attempts-1 and
    only supports a boolean exponential backoff, which the Airflow backend
    reports as a lossy mapping.
    """

    max_attempts: int = 1
    delay_seconds: float = 0.0
    backoff: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")
        if self.backoff < 1.0:
            raise ValueError("backoff must be >= 1.0")

    @property
    def retries(self) -> int:
        """Number of *re*-tries after the first attempt."""
        return self.max_attempts - 1

    def delays(self) -> list[float]:
        """Concrete delay before each retry, with backoff applied."""
        return [self.delay_seconds * (self.backoff**i) for i in range(self.retries)]


# --- input references -------------------------------------------------------


@dataclass(frozen=True)
class ParamRef:
    """A task input bound to a pipeline parameter."""

    param: str


@dataclass(frozen=True)
class OutputRef:
    """A task input bound to another task's named output."""

    task: str
    output: str = "result"


@dataclass(frozen=True)
class LiteralValue:
    """A task input bound to an inline JSON-serializable constant."""

    value: Any


InputValue = ParamRef | OutputRef | LiteralValue


# --- specs -------------------------------------------------------------------


@dataclass
class ParamSpec:
    name: str
    type: str = "Any"
    default: Any = REQUIRED
    description: str | None = None

    @property
    def required(self) -> bool:
        return self.default is REQUIRED


@dataclass
class TaskSpec:
    name: str
    fn_ref: str  # "package.module:qualname"
    inputs: dict[str, InputValue] = field(default_factory=dict)
    outputs: tuple[str, ...] = ("result",)
    output_types: dict[str, str] = field(default_factory=dict)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    resources: Resources = field(default_factory=Resources)
    cache: bool = False
    description: str | None = None

    @property
    def single_output(self) -> bool:
        return len(self.outputs) == 1

    def upstream(self) -> set[str]:
        """Names of tasks this task consumes outputs from."""
        return {v.task for v in self.inputs.values() if isinstance(v, OutputRef)}


@dataclass
class Pipeline:
    name: str
    description: str | None = None
    schedule: str | None = None
    params: dict[str, ParamSpec] = field(default_factory=dict)
    tasks: dict[str, TaskSpec] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"invalid pipeline name {self.name!r}: must match {_NAME_RE.pattern}"
            )

    def resolve_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge defaults with overrides; raise if a required param is unbound."""
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(sorted(unknown))}")
        resolved: dict[str, Any] = {}
        missing: list[str] = []
        for name, spec in self.params.items():
            if name in overrides:
                resolved[name] = overrides[name]
            elif spec.required:
                missing.append(name)
            else:
                resolved[name] = spec.default
        if missing:
            raise ValueError(f"missing required parameter(s): {', '.join(missing)}")
        return resolved
