"""End-to-end tests through the CLI entry point."""

import pytest

from hydraml.cli import main

SPEC = "examples.fraud_detection.pipeline:fraud_pipeline"


def test_backends_prints_capability_matrix(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    assert "capability" in out
    for name in ("airflow", "kubeflow", "local", "prefect"):
        assert name in out
    assert "task_caching" in out


def test_inspect_emits_ir_yaml(capsys):
    assert main(["inspect", SPEC]) == 0
    out = capsys.readouterr().out
    assert "schema_version: 1" in out
    assert "fraud-detection" in out


def test_graph_formats(capsys):
    assert main(["graph", SPEC]) == 0
    assert '"train" -> "evaluate"' in capsys.readouterr().out
    assert main(["graph", SPEC, "--format", "mermaid"]) == 0
    assert "train --> evaluate" in capsys.readouterr().out


def test_validate_all_backends(capsys):
    assert main(["validate", SPEC]) == 0
    out = capsys.readouterr().out
    assert "== airflow ==" in out and "== prefect ==" in out


def test_compile_writes_artifact(tmp_path, capsys):
    assert main(["compile", SPEC, "--backend", "prefect", "-o", str(tmp_path)]) == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("fraud_detection_prefect.py")
    assert (tmp_path / "fraud_detection_prefect.py").exists()


def test_compile_param_override_is_typed(tmp_path):
    assert (
        main(
            [
                "compile",
                SPEC,
                "--backend",
                "airflow",
                "-o",
                str(tmp_path),
                "--param",
                "epochs=12",
            ]
        )
        == 0
    )
    assert "epochs=12" in (tmp_path / "fraud_detection_airflow.py").read_text()


def test_bad_param_value_fails_cleanly(tmp_path, capsys):
    code = main(
        ["compile", SPEC, "--backend", "airflow", "-o", str(tmp_path), "--param", "epochs=lots"]
    )
    assert code == 1
    assert "cannot parse 'lots' as int" in capsys.readouterr().err


def test_unknown_param_fails_cleanly(tmp_path, capsys):
    code = main(["run", SPEC, "--param", "nope=1", "--cache-dir", str(tmp_path)])
    assert code == 1
    assert "unknown parameter" in capsys.readouterr().err


def test_run_executes_pipeline(tmp_path, capsys):
    code = main(
        [
            "run",
            SPEC,
            "--no-cache",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--param",
            "sample_rows=50",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "succeeded ingest" in out
    assert "output[evaluate]" in out


def test_missing_spec_file_fails_cleanly(capsys):
    assert main(["inspect", "does/not/exist.py"]) == 1
    assert "no such file" in capsys.readouterr().err


@pytest.mark.parametrize("args", [[], ["--version"]])
def test_help_and_version_paths(args, capsys):
    if args:
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        assert exc_info.value.code == 0
    else:
        assert main(args) == 2
