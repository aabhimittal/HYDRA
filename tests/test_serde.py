"""Round-trip tests for IR serialization."""

import pytest

from hydraml.ir.serde import from_dict, from_yaml, to_dict, to_yaml
from hydraml.loader import load_pipeline


def _example():
    return load_pipeline("examples.fraud_detection.pipeline:fraud_pipeline")


def test_yaml_round_trip_preserves_everything():
    pipe = _example()
    restored = from_yaml(to_yaml(pipe))
    assert restored == pipe


def test_dict_round_trip_preserves_everything():
    pipe = _example()
    assert from_dict(to_dict(pipe)) == pipe


def test_serialized_form_is_stable_and_explicit():
    doc = to_dict(_example())
    assert doc["schema_version"] == 1
    assert doc["schedule"] == "0 6 * * *"
    train = doc["tasks"]["train"]
    assert train["outputs"] == ["model_uri", "metrics"]
    assert train["resources"]["gpu"] == 1
    assert train["fn"] == "examples.fraud_detection.pipeline:train"
    # Required-vs-default distinction survives.
    assert "default" in doc["params"]["epochs"]


def test_unknown_schema_version_rejected():
    doc = to_dict(_example())
    doc["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        from_dict(doc)
