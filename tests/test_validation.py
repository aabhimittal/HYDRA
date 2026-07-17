"""Tests for backend capability negotiation."""

import pytest

from hydraml import Resources, RetryPolicy, pipeline, task
from hydraml.backends import get_backend
from hydraml.issues import Severity


def _codes(issues):
    return {i.code for i in issues}


@task(cache=True, retry=RetryPolicy(max_attempts=3, delay_seconds=1.0, backoff=3.0))
def cached_flaky(x: int) -> int:
    return x


@task(resources=Resources(gpu=2))
def gpu_no_type(x: int) -> int:
    return x


@pipeline(schedule="0 0 * * *")
def demanding(x: int = 1):
    gpu_no_type(x=cached_flaky(x=x))


def test_airflow_reports_lossy_mappings():
    issues = get_backend("airflow").validate(demanding)
    codes = _codes(issues)
    assert "CAP001" in codes  # compile-time params
    assert "CAP004" in codes  # backoff factor approximated
    assert "CAP005" in codes  # no caching
    assert not any(i.severity is Severity.ERROR for i in issues)


def test_kubeflow_requires_accelerator_type_for_gpus():
    issues = get_backend("kubeflow").validate(demanding)
    kf003 = [i for i in issues if i.code == "KF003"]
    assert kf003 and kf003[0].severity is Severity.ERROR
    # Providing a default accelerator resolves it.
    issues = get_backend("kubeflow", default_accelerator="NVIDIA_TESLA_T4").validate(demanding)
    assert "KF003" not in _codes(issues)


def test_prefect_flags_unenforceable_resources():
    issues = get_backend("prefect").validate(demanding)
    codes = _codes(issues)
    assert "CAP006" in codes and "CAP007" in codes
    assert "CAP004" not in codes  # arbitrary backoff is fine on Prefect


def test_local_backend_accepts_locally_defined_tasks():
    @task
    def inner(x: int) -> int:
        return x

    @pipeline
    def p(x: int = 1):
        inner(x=x)

    # fn_ref contains <locals>: codegen backends must refuse, local must not.
    airflow_issues = get_backend("airflow").validate(p)
    assert any(i.code == "CAP009" and i.severity is Severity.ERROR for i in airflow_issues)
    assert "CAP009" not in _codes(get_backend("local").validate(p))


def test_unknown_backend_and_unknown_option_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("dagster")
    with pytest.raises(ValueError, match="does not accept option"):
        get_backend("prefect", base_image="x")
