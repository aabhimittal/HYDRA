"""Tests for pipeline spec resolution."""

import textwrap

import pytest

from hydraml.errors import LoadError
from hydraml.ir.serde import to_yaml
from hydraml.loader import load_pipeline

PIPELINE_SOURCE = textwrap.dedent(
    """
    from hydraml import pipeline, task

    @task
    def one(x: int) -> int:
        return x

    @pipeline
    def alpha(x: int = 1):
        one(x=x)

    @pipeline
    def beta(x: int = 2):
        one(x=x)
    """
)


@pytest.fixture
def pipeline_file(tmp_path):
    path = tmp_path / "pipes_for_loader_test.py"
    path.write_text(PIPELINE_SOURCE)
    return path


def test_load_by_module_and_attr():
    pipe = load_pipeline("examples.fraud_detection.pipeline:fraud_pipeline")
    assert pipe.name == "fraud-detection"


def test_load_module_with_single_pipeline_needs_no_attr():
    pipe = load_pipeline("examples.fraud_detection.pipeline")
    assert pipe.name == "fraud-detection"


def test_load_by_file_path_with_attr(pipeline_file):
    pipe = load_pipeline(f"{pipeline_file}:beta")
    assert pipe.name == "beta"
    assert pipe.params["x"].default == 2


def test_ambiguous_file_without_attr_is_an_error(pipeline_file):
    with pytest.raises(LoadError, match="multiple pipelines"):
        load_pipeline(str(pipeline_file))


def test_missing_attr_and_wrong_type_errors():
    with pytest.raises(LoadError, match="no attribute"):
        load_pipeline("examples.fraud_detection.pipeline:nonexistent")
    with pytest.raises(LoadError, match="not a Pipeline"):
        load_pipeline("examples.fraud_detection.pipeline:train")


def test_load_from_serialized_yaml(tmp_path):
    pipe = load_pipeline("examples.fraud_detection.pipeline:fraud_pipeline")
    yaml_path = tmp_path / "spec.yaml"
    yaml_path.write_text(to_yaml(pipe))
    assert load_pipeline(str(yaml_path)) == pipe


def test_unimportable_module_is_a_load_error():
    with pytest.raises(LoadError, match="cannot import"):
        load_pipeline("definitely.not.a.module")
