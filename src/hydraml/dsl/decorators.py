"""The ``@task`` / ``@pipeline`` authoring API.

Tracing model: decorating a function with ``@pipeline`` runs it *once, at
definition time*, with :class:`ParamHandle` objects in place of arguments.
Task calls inside the body record :class:`~hydraml.ir.model.TaskSpec` nodes
instead of executing, and return handles. The decorator therefore evaluates
to a fully-built, immutable :class:`~hydraml.ir.model.Pipeline` IR object.

Outside a pipeline body, a ``@task``-decorated function behaves like the
plain function it wraps (call it directly in unit tests).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from ..errors import UsageError
from ..ir.model import (
    REQUIRED,
    TYPE_NAMES,
    InputValue,
    LiteralValue,
    OutputRef,
    ParamRef,
    ParamSpec,
    Pipeline,
    Resources,
    RetryPolicy,
    TaskSpec,
)
from .handles import MultiOutputHandle, OutputHandle, ParamHandle

#: Callables registered at decoration time, keyed by fn_ref. Lets the local
#: backend execute tasks without an importlib round-trip (which fails for
#: pipelines defined in scripts/REPLs).
FN_REGISTRY: dict[str, Callable[..., Any]] = {}

_ANNOTATION_MAP: dict[Any, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    dict: "dict",
    list: "list",
}


def _type_name(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    if annotation in _ANNOTATION_MAP:
        return _ANNOTATION_MAP[annotation]
    # Accept string annotations ("str", "dict[str, float]", ...) leniently.
    if isinstance(annotation, str):
        base = annotation.split("[", 1)[0].strip().lower()
        return base if base in TYPE_NAMES else "Any"
    origin = getattr(annotation, "__origin__", None)
    if origin in _ANNOTATION_MAP:
        return _ANNOTATION_MAP[origin]
    return "Any"


# --- build context ------------------------------------------------------------

_BUILD_STACK: list[_PipelineBuilder] = []


class _PipelineBuilder:
    def __init__(self) -> None:
        self.tasks: dict[str, TaskSpec] = {}
        self._name_counts: dict[str, int] = {}

    def unique_name(self, base: str) -> str:
        count = self._name_counts.get(base, 0) + 1
        self._name_counts[base] = count
        return base if count == 1 else f"{base}_{count}"

    def add(self, spec: TaskSpec) -> None:
        self.tasks[spec.name] = spec


def _current_builder() -> _PipelineBuilder | None:
    return _BUILD_STACK[-1] if _BUILD_STACK else None


# --- @task ---------------------------------------------------------------------


class TaskDefinition:
    """Wraps a plain function with orchestration metadata.

    Calling it inside a pipeline body records an IR node; calling it anywhere
    else runs the underlying function directly.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        outputs: tuple[str, ...] | dict[str, Any] | None = None,
        retry: RetryPolicy | None = None,
        resources: Resources | None = None,
        cache: bool = False,
        description: str | None = None,
    ):
        sig = inspect.signature(fn)
        for param in sig.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise UsageError(
                    f"task {fn.__name__!r}: *args/**kwargs signatures cannot be"
                    " represented in the IR; declare explicit parameters"
                )
        self.fn = fn
        self.name = name or fn.__name__
        self.retry = retry or RetryPolicy()
        self.resources = resources or Resources()
        self.cache = cache
        self.description = description or _first_line(fn.__doc__)
        self._signature = sig

        if outputs is None:
            self.outputs: tuple[str, ...] = ("result",)
            self.output_types: dict[str, str] = {"result": _type_name(sig.return_annotation)}
        elif isinstance(outputs, dict):
            self.outputs = tuple(outputs)
            self.output_types = {k: _type_name(v) for k, v in outputs.items()}
        else:
            self.outputs = tuple(outputs)
            self.output_types = {k: "Any" for k in self.outputs}
        if not self.outputs:
            raise UsageError(f"task {self.name!r}: outputs cannot be empty")

        self.fn_ref = f"{fn.__module__}:{fn.__qualname__}"
        FN_REGISTRY[self.fn_ref] = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        builder = _current_builder()
        if builder is None:
            return self.fn(*args, **kwargs)
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise UsageError(f"task {self.name!r}: {exc}") from exc

        task_name = builder.unique_name(self.name)
        inputs = {
            arg: _to_input_value(self.name, arg, value)
            for arg, value in bound.arguments.items()
        }
        builder.add(
            TaskSpec(
                name=task_name,
                fn_ref=self.fn_ref,
                inputs=inputs,
                outputs=self.outputs,
                output_types=dict(self.output_types),
                retry=self.retry,
                resources=self.resources,
                cache=self.cache,
                description=self.description,
            )
        )
        if len(self.outputs) == 1:
            return OutputHandle(task_name, self.outputs[0])
        return MultiOutputHandle(task_name, self.outputs)

    def __repr__(self) -> str:
        return f"<hydra task {self.name!r} ({self.fn_ref})>"


def _to_input_value(task: str, arg: str, value: Any) -> InputValue:
    if isinstance(value, OutputHandle):
        return OutputRef(task=value.task, output=value.output)
    if isinstance(value, ParamHandle):
        return ParamRef(param=value.param)
    if isinstance(value, MultiOutputHandle):
        raise UsageError(
            f"task {task!r}, input {arg!r}: task {value.task!r} has multiple"
            f" outputs ({', '.join(value.outputs)}); select one, e.g."
            f" handle[{value.outputs[0]!r}]"
        )
    if isinstance(value, TaskDefinition):
        raise UsageError(
            f"task {task!r}, input {arg!r}: got a task definition instead of a"
            " value - did you forget to call it?"
        )
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise UsageError(
            f"task {task!r}, input {arg!r}: literal of type"
            f" {type(value).__name__} is not JSON-serializable. Values that"
            " cross task boundaries must be small and serializable; pass"
            " heavy objects by reference (URI)."
        ) from exc
    return LiteralValue(value=value)


def task(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    outputs: tuple[str, ...] | dict[str, Any] | None = None,
    retry: RetryPolicy | None = None,
    resources: Resources | None = None,
    cache: bool = False,
    description: str | None = None,
) -> Any:
    """Declare a function as a HYDRA task. Usable bare or with arguments."""

    def wrap(f: Callable[..., Any]) -> TaskDefinition:
        return TaskDefinition(
            f,
            name=name,
            outputs=outputs,
            retry=retry,
            resources=resources,
            cache=cache,
            description=description,
        )

    return wrap(fn) if fn is not None else wrap


# --- @pipeline -------------------------------------------------------------------


def pipeline(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schedule: str | None = None,
    tags: tuple[str, ...] = (),
) -> Any:
    """Declare a pipeline. The decorated function is traced immediately and the
    decorator evaluates to a :class:`~hydraml.ir.model.Pipeline` IR object."""

    def wrap(f: Callable[..., Any]) -> Pipeline:
        return _trace(f, name=name, description=description, schedule=schedule, tags=tags)

    return wrap(fn) if fn is not None else wrap


def _trace(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    schedule: str | None,
    tags: tuple[str, ...],
) -> Pipeline:
    sig = inspect.signature(fn)
    params: dict[str, ParamSpec] = {}
    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise UsageError(
                f"pipeline {fn.__name__!r}: *args/**kwargs parameters are not supported"
            )
        default = REQUIRED if param.default is inspect.Parameter.empty else param.default
        if default is not REQUIRED:
            try:
                json.dumps(default)
            except (TypeError, ValueError) as exc:
                raise UsageError(
                    f"pipeline {fn.__name__!r}: default for {pname!r} is not"
                    " JSON-serializable"
                ) from exc
        params[pname] = ParamSpec(name=pname, type=_type_name(param.annotation), default=default)

    builder = _PipelineBuilder()
    _BUILD_STACK.append(builder)
    try:
        fn(**{pname: ParamHandle(pname) for pname in params})
    finally:
        _BUILD_STACK.pop()

    if not builder.tasks:
        raise UsageError(f"pipeline {fn.__name__!r} defines no tasks")

    return Pipeline(
        name=name or fn.__name__,
        description=description or _first_line(fn.__doc__),
        schedule=schedule,
        params=params,
        tasks=builder.tasks,
        tags=tuple(tags),
    )


def _first_line(doc: str | None) -> str | None:
    if not doc:
        return None
    return doc.strip().splitlines()[0].strip() or None
