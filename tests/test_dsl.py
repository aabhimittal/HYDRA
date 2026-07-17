"""Tests for the @task / @pipeline tracing DSL."""

import pytest

from hydraml import Resources, RetryPolicy, pipeline, task
from hydraml.dsl.handles import OutputHandle
from hydraml.errors import UsageError
from hydraml.ir.model import LiteralValue, OutputRef, ParamRef


@task
def add(a: int, b: int) -> int:
    return a + b


@task(outputs={"quotient": int, "remainder": int})
def divmod_task(a: int, b: int) -> dict:
    q, r = divmod(a, b)
    return {"quotient": q, "remainder": r}


def test_task_is_a_plain_function_outside_pipelines():
    assert add(2, 3) == 5
    assert divmod_task(7, 2) == {"quotient": 3, "remainder": 1}


def test_tracing_builds_ir():
    @pipeline
    def p(x: int = 1):
        """Adds things."""
        total = add(a=x, b=2)
        add(a=total, b=3)

    assert p.name == "p"
    assert p.description == "Adds things."
    assert list(p.tasks) == ["add", "add_2"]
    assert p.params["x"].type == "int"
    assert p.params["x"].default == 1

    first = p.tasks["add"]
    assert first.inputs["a"] == ParamRef("x")
    assert first.inputs["b"] == LiteralValue(2)
    second = p.tasks["add_2"]
    assert second.inputs["a"] == OutputRef("add", "result")


def test_positional_arguments_map_to_parameter_names():
    @pipeline
    def p():
        add(1, 2)

    assert p.tasks["add"].inputs == {"a": LiteralValue(1), "b": LiteralValue(2)}


def test_multi_output_selection():
    @pipeline
    def p():
        parts = divmod_task(a=7, b=2)
        add(a=parts["quotient"], b=parts.remainder)

    spec = p.tasks["add"]
    assert spec.inputs["a"] == OutputRef("divmod_task", "quotient")
    assert spec.inputs["b"] == OutputRef("divmod_task", "remainder")
    assert p.tasks["divmod_task"].output_types == {"quotient": "int", "remainder": "int"}


def test_multi_output_handle_rejects_unknown_output():
    with pytest.raises(UsageError, match="no output"):

        @pipeline
        def p():
            parts = divmod_task(a=7, b=2)
            add(a=parts["nope"], b=1)


def test_passing_whole_multi_output_handle_is_an_error():
    with pytest.raises(UsageError, match="select one"):

        @pipeline
        def p():
            add(a=divmod_task(a=7, b=2), b=1)


def test_handles_cannot_be_used_as_values():
    with pytest.raises(UsageError, match="string-format"):

        @pipeline
        def p(x: str = "a"):
            add(a=f"prefix-{x}", b=1)


def test_non_json_literal_rejected():
    with pytest.raises(UsageError, match="JSON-serializable"):

        @pipeline
        def p():
            add(a=object(), b=1)


def test_uncalled_task_passed_as_input_has_helpful_error():
    with pytest.raises(UsageError, match="forget to call"):

        @pipeline
        def p():
            add(a=add, b=1)


def test_pipeline_without_tasks_rejected():
    with pytest.raises(UsageError, match="defines no tasks"):

        @pipeline
        def p():
            pass


def test_var_args_signatures_rejected():
    with pytest.raises(UsageError, match=r"\*args"):

        @task
        def bad(*args):
            pass


def test_task_metadata_lands_in_ir():
    @task(
        name="renamed",
        retry=RetryPolicy(max_attempts=3, delay_seconds=1.0, backoff=2.0),
        resources=Resources(cpu="1", gpu=2, gpu_type="A100"),
        cache=True,
    )
    def heavy(x: int) -> str:
        return str(x)

    @pipeline(name="meta", schedule="0 0 * * *", tags=("a",))
    def p():
        heavy(x=1)

    spec = p.tasks["renamed"]
    assert spec.retry.retries == 2
    assert spec.retry.delays() == [1.0, 2.0]
    assert spec.resources.gpu_type == "A100"
    assert spec.cache is True
    assert p.schedule == "0 0 * * *"
    assert p.tags == ("a",)


def test_required_params_have_no_default():
    @pipeline
    def p(required_one: str, optional: int = 5):
        add(a=optional, b=1)

    assert p.params["required_one"].required
    assert not p.params["optional"].required
    with pytest.raises(ValueError, match="required"):
        p.resolve_params({})
    assert p.resolve_params({"required_one": "x"})["optional"] == 5


def test_output_handle_is_returned_for_single_output():
    @pipeline
    def p():
        handle = add(a=1, b=2)
        assert isinstance(handle, OutputHandle)
        add(a=handle, b=3)
