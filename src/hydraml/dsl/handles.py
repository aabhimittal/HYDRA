"""Symbolic handles returned while tracing a pipeline function.

Inside a ``@pipeline`` body, task calls do not execute - they return handles
that stand for future values. Handles may only be passed onward into other
task calls; any attempt to *use* one (string-format it, add to it, branch on
it) raises immediately with an explanation, because that computation would
happen at trace time, not at run time on the backend.
"""

from __future__ import annotations

from ..errors import UsageError


class _Symbolic:
    """Shared guard rails for all handle types."""

    def _describe(self) -> str:  # pragma: no cover - overridden
        return "a symbolic value"

    def _forbid(self, operation: str) -> UsageError:
        return UsageError(
            f"cannot {operation} {self._describe()} inside a pipeline definition: "
            "the value does not exist until the pipeline runs on a backend. "
            "Move this logic inside a @task function."
        )

    def __str__(self) -> str:
        raise self._forbid("string-format")

    def __bool__(self) -> bool:
        raise self._forbid("branch on")

    def __iter__(self):
        raise self._forbid("iterate over")

    def __add__(self, other):
        raise self._forbid("do arithmetic on")

    __radd__ = __sub__ = __mul__ = __truediv__ = __add__


class ParamHandle(_Symbolic):
    __slots__ = ("param",)

    def __init__(self, param: str):
        self.param = param

    def _describe(self) -> str:
        return f"pipeline parameter {self.param!r}"

    def __repr__(self) -> str:
        return f"ParamHandle({self.param!r})"


class OutputHandle(_Symbolic):
    __slots__ = ("task", "output")

    def __init__(self, task: str, output: str = "result"):
        self.task = task
        self.output = output

    def _describe(self) -> str:
        return f"output {self.output!r} of task {self.task!r}"

    def __repr__(self) -> str:
        return f"OutputHandle({self.task!r}, {self.output!r})"


class MultiOutputHandle(_Symbolic):
    """Handle for a task declaring several named outputs; index to select one."""

    __slots__ = ("task", "outputs")

    def __init__(self, task: str, outputs: tuple[str, ...]):
        self.task = task
        self.outputs = outputs

    def _describe(self) -> str:
        return f"multi-output task {self.task!r}"

    def __getitem__(self, key: str) -> OutputHandle:
        if key not in self.outputs:
            raise UsageError(
                f"task {self.task!r} has no output {key!r}"
                f" (declared outputs: {', '.join(self.outputs)})"
            )
        return OutputHandle(self.task, key)

    def __getattr__(self, key: str) -> OutputHandle:
        if key.startswith("_"):
            raise AttributeError(key)
        return self[key]

    def __repr__(self) -> str:
        return f"MultiOutputHandle({self.task!r}, outputs={self.outputs!r})"
