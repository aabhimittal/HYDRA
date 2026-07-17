"""Codegen tests: golden files plus structural checks on generated source.

The goldens under tests/golden/ are the compiled example pipeline and double
as the codegen spec: a diff here should be a deliberate, reviewed change to
what HYDRA emits.
"""

from pathlib import Path

import pytest

from hydraml.backends import get_backend
from hydraml.errors import CompilationError
from hydraml.loader import load_pipeline

GOLDEN_DIR = Path(__file__).parent / "golden"

CASES = [
    ("airflow", {}, "fraud_detection_airflow.py"),
    ("prefect", {}, "fraud_detection_prefect.py"),
    ("kubeflow", {}, "fraud_detection_kubeflow.py"),
]


def _example():
    return load_pipeline("examples.fraud_detection.pipeline:fraud_pipeline")


@pytest.mark.parametrize(("backend_name", "opts", "filename"), CASES)
def test_generated_code_matches_golden(tmp_path, backend_name, opts, filename):
    result = get_backend(backend_name, **opts).compile(_example(), tmp_path)
    (generated,) = result.files
    assert generated.name == filename
    expected = (GOLDEN_DIR / filename).read_text()
    assert generated.read_text() == expected


@pytest.mark.parametrize(("backend_name", "opts", "filename"), CASES)
def test_generated_code_is_valid_python(tmp_path, backend_name, opts, filename):
    result = get_backend(backend_name, **opts).compile(_example(), tmp_path)
    source = result.files[0].read_text()
    compile(source, filename, "exec")  # SyntaxError -> fail


def test_compile_time_param_binding_for_airflow(tmp_path):
    result = get_backend("airflow").compile(
        _example(), tmp_path, params={"learning_rate": 0.5, "epochs": 20}
    )
    source = result.files[0].read_text()
    assert "learning_rate=0.5" in source
    assert "epochs=20" in source
    assert "learning_rate=0.01" not in source


def test_runtime_param_backends_keep_defaults_in_signature(tmp_path):
    source = get_backend("prefect").compile(
        _example(), tmp_path, params={"epochs": 20}
    ).files[0].read_text()
    assert "epochs: int = 20" in source  # override becomes the new default


def test_kubeflow_backend_options_flow_into_components(tmp_path):
    source = get_backend(
        "kubeflow", base_image="ghcr.io/acme/train:1", packages="hydraml,scikit-learn"
    ).compile(_example(), tmp_path).files[0].read_text()
    assert "base_image='ghcr.io/acme/train:1'" in source
    assert "packages_to_install=['hydraml', 'scikit-learn']" in source


def test_kubeflow_normalizes_caching_defaults(tmp_path):
    source = get_backend("kubeflow").compile(_example(), tmp_path).files[0].read_text()
    # KFP caches by default; HYDRA emits an explicit choice for every task.
    assert source.count("set_caching_options(") == 4
    assert "featurize_task.set_caching_options(True)" in source
    assert "ingest_task.set_caching_options(False)" in source


def test_validation_errors_block_compilation(tmp_path):
    import copy

    from hydraml.ir.model import Resources

    # Deep-copy: load_pipeline returns the module-cached Pipeline object.
    pipe = copy.deepcopy(_example())
    pipe.tasks["train"].resources = Resources(gpu=1)  # no gpu_type
    with pytest.raises(CompilationError) as exc_info:
        get_backend("kubeflow").compile(pipe, tmp_path)
    assert any(i.code == "KF003" for i in exc_info.value.issues)
    assert not list(tmp_path.iterdir())  # nothing half-written


def test_local_backend_refuses_to_compile(tmp_path):
    with pytest.raises(CompilationError, match="hydraml run"):
        get_backend("local").compile(_example(), tmp_path)
