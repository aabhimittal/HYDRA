"""Tests for the in-process execution backend."""

import pytest

from hydraml import RetryPolicy, pipeline, task
from hydraml.backends.local import LocalBackend

CALLS = {"flaky": 0, "counted": 0}


@task(retry=RetryPolicy(max_attempts=3, delay_seconds=2.0, backoff=2.0))
def flaky(fail_times: int) -> str:
    CALLS["flaky"] += 1
    if CALLS["flaky"] <= fail_times:
        raise RuntimeError("transient")
    return "ok"


@task(cache=True)
def counted(x: int) -> int:
    CALLS["counted"] += 1
    return x * 10


@task(outputs=("a", "b"))
def two_outputs(good: bool) -> dict:
    return {"a": 1, "b": 2} if good else {"a": 1}


@task
def consume(value: int) -> int:
    return value + 1


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS["flaky"] = 0
    CALLS["counted"] = 0


def _backend(tmp_path):
    return LocalBackend(cache_dir=str(tmp_path / "cache"))


def test_retries_follow_the_backoff_schedule(tmp_path):
    @pipeline
    def p():
        flaky(fail_times=2)

    slept: list[float] = []
    result = _backend(tmp_path).run(p, sleeper=slept.append)
    run = result.task_runs["flaky"]
    assert run.status == "succeeded"
    assert run.attempts == 3
    assert slept == [2.0, 4.0]  # delay * backoff^n


def test_retry_budget_exhaustion_fails_and_skips_downstream(tmp_path):
    @pipeline
    def p():
        value = flaky(fail_times=99)
        consume(value=value)

    result = _backend(tmp_path).run(p, sleeper=lambda _: None)
    assert not result.succeeded
    assert result.task_runs["flaky"].status == "failed"
    assert "transient" in result.task_runs["flaky"].error
    assert result.task_runs["consume"].status == "skipped"


def test_caching_skips_execution_on_identical_inputs(tmp_path):
    @pipeline
    def p(x: int = 3):
        counted(x=x)

    backend = _backend(tmp_path)
    first = backend.run(p)
    second = backend.run(p)
    changed = backend.run(p, params={"x": 4})

    assert first.task_runs["counted"].status == "succeeded"
    assert second.task_runs["counted"].status == "cached"
    assert second.outputs["counted"] == 30
    assert changed.task_runs["counted"].status == "succeeded"
    assert CALLS["counted"] == 2  # only the two distinct inputs executed


def test_no_cache_flag_forces_execution(tmp_path):
    @pipeline
    def p():
        counted(x=1)

    backend = _backend(tmp_path)
    backend.run(p)
    backend.run(p, use_cache=False)
    assert CALLS["counted"] == 2


def test_multi_output_routing(tmp_path):
    @pipeline
    def p():
        parts = two_outputs(good=True)
        consume(value=parts["b"])

    result = _backend(tmp_path).run(p)
    assert result.succeeded
    assert result.outputs["consume"] == 3


def test_output_contract_violation_fails_without_retry(tmp_path):
    @task(outputs=("a", "b"), retry=RetryPolicy(max_attempts=5, delay_seconds=1.0))
    def broken() -> dict:
        return {"a": 1}  # missing "b"

    @pipeline
    def p():
        broken()

    slept: list[float] = []
    result = _backend(tmp_path).run(p, sleeper=slept.append)
    run = result.task_runs["broken"]
    assert run.status == "failed"
    assert "missing declared output" in run.error
    assert run.attempts == 1  # contract violations are not transient
    assert slept == []


def test_param_overrides_flow_into_tasks(tmp_path):
    @pipeline
    def p(x: int = 1):
        consume(value=x)

    result = _backend(tmp_path).run(p, params={"x": 41})
    assert result.outputs["consume"] == 42
