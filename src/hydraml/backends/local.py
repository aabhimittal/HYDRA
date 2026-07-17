"""Local backend: executes the IR in-process.

This is not a toy - it is the reference implementation of HYDRA's execution
semantics (retry schedule, opt-in content-addressed caching, pass-by-value of
small results) and the fastest way to iterate on a pipeline before compiling
it for a real orchestrator. Resources are validated but not enforced.

Cache key = SHA-256 of (fn_ref, function source, resolved inputs). Results are
stored as JSON, which doubles as an enforcement of the "small, serializable
values only" tenet: non-serializable results simply skip the cache.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dsl.decorators import FN_REGISTRY, TaskDefinition
from ..errors import CompilationError, ExecutionError, HydraError
from ..ir.graph import topological_order
from ..ir.model import LiteralValue, OutputRef, ParamRef, Pipeline, TaskSpec
from ..issues import errors
from .base import Backend, Capability, register


@dataclass
class TaskRun:
    task: str
    status: str  # "succeeded" | "cached" | "failed" | "skipped"
    attempts: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class RunResult:
    pipeline: str
    params: dict[str, Any]
    task_runs: dict[str, TaskRun] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)  # task -> return value

    @property
    def succeeded(self) -> bool:
        return all(r.status in ("succeeded", "cached") for r in self.task_runs.values())


@register
class LocalBackend(Backend):
    name = "local"
    capabilities = frozenset(
        {
            Capability.RUNTIME_PARAMS,
            Capability.RETRIES,
            Capability.RETRY_BACKOFF_FACTOR,
            Capability.TASK_CACHING,
            Capability.MULTI_OUTPUT,
        }
    )
    # The local runner resolves callables through FN_REGISTRY, so script-local
    # and REPL-defined tasks are fine here.
    requires_importable_functions = False

    @classmethod
    def option_names(cls) -> tuple[str, ...]:
        return ("cache_dir",)

    def emit(self, pipeline: Pipeline, outdir: Path, params: dict[str, Any]) -> list[Path]:
        raise CompilationError(
            "the local backend executes pipelines directly; use `hydraml run`"
            " instead of `hydraml compile --backend local`"
        )

    # --- execution ---------------------------------------------------------------

    def run(
        self,
        pipeline: Pipeline,
        params: dict[str, Any] | None = None,
        *,
        use_cache: bool = True,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> RunResult:
        issues = self.validate(pipeline)
        if errors(issues):
            raise CompilationError(
                f"cannot run pipeline {pipeline.name!r}: validation failed", issues=issues
            )
        resolved = pipeline.resolve_params(params)
        result = RunResult(pipeline=pipeline.name, params=resolved)
        cache = _Cache(Path(self.options.get("cache_dir", ".hydra_cache")))

        failed_upstream: set[str] = set()
        for tname in topological_order(pipeline):
            tspec = pipeline.tasks[tname]
            if tspec.upstream() & failed_upstream:
                result.task_runs[tname] = TaskRun(task=tname, status="skipped")
                failed_upstream.add(tname)
                continue
            run = self._run_task(
                pipeline, tspec, resolved, result.outputs, cache, use_cache, sleeper
            )
            result.task_runs[tname] = run
            if run.status == "failed":
                failed_upstream.add(tname)
        return result

    def _run_task(
        self,
        pipeline: Pipeline,
        tspec: TaskSpec,
        params: dict[str, Any],
        outputs: dict[str, Any],
        cache: _Cache,
        use_cache: bool,
        sleeper: Callable[[float], None],
    ) -> TaskRun:
        kwargs = {
            arg: _resolve_input(value, params, outputs, pipeline)
            for arg, value in tspec.inputs.items()
        }
        fn = resolve_callable(tspec.fn_ref)

        key = cache.key(tspec, fn, kwargs) if (use_cache and tspec.cache) else None
        if key is not None:
            hit, value = cache.get(key)
            if hit:
                outputs[tspec.name] = value
                return TaskRun(task=tspec.name, status="cached")

        run = TaskRun(task=tspec.name, status="failed")
        started = time.monotonic()
        delays = [0.0, *tspec.retry.delays()]
        last_error: BaseException | None = None
        for attempt, delay in enumerate(delays, start=1):
            if delay:
                sleeper(delay)
            run.attempts = attempt
            try:
                value = fn(**kwargs)
            except Exception as exc:  # noqa: BLE001 - user task code
                last_error = exc
                continue
            try:
                _check_outputs(tspec, value)
            except ExecutionError as exc:
                # Contract violation, not a transient failure: don't retry.
                last_error = exc.cause
                break
            outputs[tspec.name] = value
            run.status = "succeeded"
            run.duration_seconds = time.monotonic() - started
            if key is not None:
                cache.put(key, value)
            return run
        run.duration_seconds = time.monotonic() - started
        run.error = repr(last_error)
        return run


def _resolve_input(
    value: ParamRef | OutputRef | LiteralValue,
    params: dict[str, Any],
    outputs: dict[str, Any],
    pipeline: Pipeline,
) -> Any:
    if isinstance(value, LiteralValue):
        return value.value
    if isinstance(value, ParamRef):
        return params[value.param]
    upstream_value = outputs[value.task]
    if pipeline.tasks[value.task].single_output:
        return upstream_value
    return upstream_value[value.output]


def _check_outputs(tspec: TaskSpec, value: Any) -> None:
    if tspec.single_output:
        return
    if not isinstance(value, dict):
        raise ExecutionError(
            tspec.name,
            TypeError(
                f"task declares outputs {tspec.outputs!r} and must return a"
                f" dict with those keys, got {type(value).__name__}"
            ),
        )
    missing = set(tspec.outputs) - set(value)
    if missing:
        raise ExecutionError(
            tspec.name,
            KeyError(f"task result is missing declared output(s): {', '.join(sorted(missing))}"),
        )


def resolve_callable(fn_ref: str) -> Callable[..., Any]:
    """Resolve a `module:qualname` reference to the underlying task function."""
    if fn_ref in FN_REGISTRY:
        return FN_REGISTRY[fn_ref]
    module_name, _, qualname = fn_ref.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise HydraError(f"cannot import module for task function {fn_ref!r}: {exc}") from exc
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if isinstance(obj, TaskDefinition):
        return obj.fn
    if callable(obj):
        return obj
    raise HydraError(f"{fn_ref!r} does not resolve to a callable")


class _Cache:
    def __init__(self, root: Path):
        self.root = root

    def key(self, tspec: TaskSpec, fn: Callable[..., Any], kwargs: dict[str, Any]) -> str | None:
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            source = getattr(getattr(fn, "__code__", None), "co_code", b"").hex()
        try:
            payload = json.dumps(
                {"fn": tspec.fn_ref, "source": source, "inputs": kwargs}, sort_keys=True
            )
        except (TypeError, ValueError):
            return None  # unserializable inputs -> uncacheable
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> tuple[bool, Any]:
        path = self._path(key)
        if not path.exists():
            return False, None
        try:
            return True, json.loads(path.read_text())["value"]
        except (ValueError, KeyError, OSError):
            return False, None

    def put(self, key: str, value: Any) -> None:
        try:
            payload = json.dumps({"value": value})
        except (TypeError, ValueError):
            return  # unserializable result -> skip cache, never fail the run
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(payload)
