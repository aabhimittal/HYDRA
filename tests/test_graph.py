"""Tests for graph algorithms and structural validation."""

import pytest

from hydraml.errors import CycleError
from hydraml.ir.graph import structural_issues, to_dot, to_mermaid, topological_order
from hydraml.ir.model import (
    LiteralValue,
    OutputRef,
    ParamRef,
    ParamSpec,
    Pipeline,
    TaskSpec,
)
from hydraml.issues import Severity


def _task(name, inputs=None, outputs=("result",)):
    return TaskSpec(name=name, fn_ref=f"mod:{name}", inputs=inputs or {}, outputs=outputs)


def test_topological_order_is_stable_and_valid():
    pipe = Pipeline(
        name="p",
        tasks={
            "c": _task("c", {"x": OutputRef("a"), "y": OutputRef("b")}),
            "b": _task("b", {"x": OutputRef("a")}),
            "a": _task("a"),
            "d": _task("d"),
        },
    )
    order = topological_order(pipe)
    assert order.index("a") < order.index("b") < order.index("c")
    # Independent roots keep definition order: 'a' was defined after 'd'... it
    # wasn't - 'a' is defined third, 'd' fourth, both depth 0.
    assert order.index("a") < order.index("d")


def test_cycle_detection():
    pipe = Pipeline(
        name="p",
        tasks={
            "a": _task("a", {"x": OutputRef("b")}),
            "b": _task("b", {"x": OutputRef("a")}),
        },
    )
    with pytest.raises(CycleError):
        topological_order(pipe)
    codes = {i.code for i in structural_issues(pipe)}
    assert "IR009" in codes


def test_structural_issues_catch_dangling_references():
    pipe = Pipeline(
        name="p",
        params={"known": ParamSpec(name="known", type="str", default="v")},
        tasks={
            "t": _task(
                "t",
                {
                    "a": ParamRef("unknown_param"),
                    "b": OutputRef("unknown_task"),
                    "c": LiteralValue(object()),
                },
            ),
            "u": _task("u", {"x": OutputRef("t", "not_an_output")}),
        },
    )
    issues = structural_issues(pipe)
    codes = sorted(i.code for i in issues)
    assert codes == ["IR004", "IR005", "IR006", "IR008"]
    assert all(i.severity is Severity.ERROR for i in issues)


def test_self_dependency_reported():
    pipe = Pipeline(name="p", tasks={"t": _task("t", {"x": OutputRef("t")})})
    codes = {i.code for i in structural_issues(pipe)}
    assert "IR007" in codes


def test_clean_pipeline_has_no_structural_issues():
    pipe = Pipeline(
        name="p",
        params={"x": ParamSpec(name="x", type="int", default=3)},
        tasks={
            "a": _task("a", {"v": ParamRef("x")}),
            "b": _task("b", {"v": OutputRef("a")}),
        },
    )
    assert structural_issues(pipe) == []


def test_graph_exports_contain_edges():
    pipe = Pipeline(
        name="p",
        tasks={"a": _task("a"), "b": _task("b", {"x": OutputRef("a")})},
    )
    dot = to_dot(pipe)
    assert '"a" -> "b"' in dot
    mermaid = to_mermaid(pipe)
    assert "a --> b" in mermaid
