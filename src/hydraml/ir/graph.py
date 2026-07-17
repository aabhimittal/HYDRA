"""Graph algorithms and structural validation over the IR.

Backend-agnostic checks live here; capability checks (what a *specific*
backend can honor) live in :mod:`hydraml.backends.base`.
"""

from __future__ import annotations

from graphlib import CycleError as _StdCycleError
from graphlib import TopologicalSorter

from ..errors import CycleError
from ..issues import Issue, Severity
from .model import TYPE_NAMES, LiteralValue, OutputRef, ParamRef, Pipeline

_JSON_TYPES = (str, int, float, bool, type(None), dict, list, tuple)


def topological_order(pipeline: Pipeline) -> list[str]:
    """Deterministic topological order (stable w.r.t. task definition order)."""
    sorter: TopologicalSorter[str] = TopologicalSorter()
    for name, task in pipeline.tasks.items():
        # Dangling references are IR005's job; ordering ignores them.
        sorter.add(name, *sorted(task.upstream() & set(pipeline.tasks)))
    try:
        order = list(sorter.static_order())
    except _StdCycleError as exc:
        # graphlib reports the cycle as the second arg: [a, ..., a]
        cycle = list(exc.args[1])[:-1] if len(exc.args) > 1 else []
        raise CycleError(cycle) from exc
    # static_order is topological but not definition-ordered among independent
    # tasks; re-sort ties by definition index for reproducible codegen.
    index = {name: i for i, name in enumerate(pipeline.tasks)}
    depth: dict[str, int] = {}
    for name in order:
        ups = pipeline.tasks[name].upstream() & set(pipeline.tasks)
        depth[name] = 1 + max((depth[u] for u in ups), default=-1)
    return sorted(order, key=lambda n: (depth[n], index[n]))


def structural_issues(pipeline: Pipeline) -> list[Issue]:
    """Backend-independent integrity checks on the IR."""
    issues: list[Issue] = []

    for pname, pspec in pipeline.params.items():
        if pspec.type not in TYPE_NAMES:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "IR001",
                    f"param {pname!r} has unknown type {pspec.type!r}"
                    f" (expected one of {', '.join(TYPE_NAMES)})",
                )
            )
        if not pspec.required and not _json_like(pspec.default):
            issues.append(
                Issue(
                    Severity.ERROR,
                    "IR002",
                    f"param {pname!r} default is not JSON-serializable:"
                    f" {type(pspec.default).__name__}",
                )
            )

    for tname, task in pipeline.tasks.items():
        if len(set(task.outputs)) != len(task.outputs):
            issues.append(
                Issue(Severity.ERROR, "IR003", "duplicate output names", task=tname)
            )
        for iname, value in task.inputs.items():
            if isinstance(value, ParamRef) and value.param not in pipeline.params:
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "IR004",
                        f"input {iname!r} references unknown param {value.param!r}",
                        task=tname,
                    )
                )
            elif isinstance(value, OutputRef):
                upstream = pipeline.tasks.get(value.task)
                if upstream is None:
                    issues.append(
                        Issue(
                            Severity.ERROR,
                            "IR005",
                            f"input {iname!r} references unknown task {value.task!r}",
                            task=tname,
                        )
                    )
                elif value.output not in upstream.outputs:
                    issues.append(
                        Issue(
                            Severity.ERROR,
                            "IR006",
                            f"input {iname!r} references unknown output"
                            f" {value.task!r}.{value.output!r}"
                            f" (has: {', '.join(upstream.outputs)})",
                            task=tname,
                        )
                    )
                if value.task == tname:
                    issues.append(
                        Issue(Severity.ERROR, "IR007", "task depends on itself", task=tname)
                    )
            elif isinstance(value, LiteralValue) and not _json_like(value.value):
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "IR008",
                        f"input {iname!r} literal is not JSON-serializable:"
                        f" {type(value.value).__name__}",
                        task=tname,
                    )
                )

    try:
        topological_order(pipeline)
    except CycleError as exc:
        issues.append(
            Issue(
                Severity.ERROR,
                "IR009",
                "cycle detected: " + " -> ".join([*exc.cycle, exc.cycle[0]] if exc.cycle else []),
            )
        )
    return issues


def _json_like(value: object) -> bool:
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_like(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_like(v) for v in value)
    return isinstance(value, _JSON_TYPES) or value is None


# --- graph export ------------------------------------------------------------


def to_dot(pipeline: Pipeline) -> str:
    lines = [f'digraph "{pipeline.name}" {{', "  rankdir=LR;", "  node [shape=box];"]
    for name in pipeline.tasks:
        lines.append(f'  "{name}";')
    for name, task in pipeline.tasks.items():
        for value in task.inputs.values():
            if isinstance(value, OutputRef):
                label = "" if task.single_output else f' [label="{value.output}"]'
                lines.append(f'  "{value.task}" -> "{name}"{label};')
    lines.append("}")
    return "\n".join(lines) + "\n"


def to_mermaid(pipeline: Pipeline) -> str:
    lines = ["flowchart LR"]
    edges = set()
    for name, task in pipeline.tasks.items():
        for value in task.inputs.values():
            if isinstance(value, OutputRef):
                edges.add((value.task, name))
    for src, dst in sorted(edges):
        lines.append(f"    {src} --> {dst}")
    for name in pipeline.tasks:
        if not any(name in edge for edge in edges):
            lines.append(f"    {name}")
    return "\n".join(lines) + "\n"
