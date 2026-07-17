"""Example: a fraud-detection training pipeline.

Pure-stdlib task bodies so the pipeline runs anywhere with
``hydraml run examples/fraud_detection/pipeline.py`` - swap the internals for
real feature stores / trainers without touching the orchestration structure.

Note how tasks exchange *references and small metrics*, not dataframes: the
featurize step returns a URI, and train returns a model URI plus a metrics
dict. That is the HYDRA data-passing contract, and it is what makes the same
definition compilable to XCom (Airflow), return values (Prefect), and
component parameters (Kubeflow).
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from pathlib import Path

from hydraml import Resources, RetryPolicy, pipeline, task


def _workdir() -> Path:
    path = Path(tempfile.gettempdir()) / "hydra_fraud_demo"
    path.mkdir(exist_ok=True)
    return path


@task(retry=RetryPolicy(max_attempts=3, delay_seconds=5.0, backoff=2.0))
def ingest(raw_path: str, sample_rows: int) -> str:
    """Pull raw transactions to a local staging file (simulated)."""
    rng = random.Random(42)
    rows = [
        {
            "amount": round(rng.lognormvariate(3.0, 1.2), 2),
            "hour": rng.randint(0, 23),
            "is_fraud": int(rng.random() < 0.02),
        }
        for _ in range(sample_rows)
    ]
    out = _workdir() / "transactions.json"
    out.write_text(json.dumps({"source": raw_path, "rows": rows}))
    return str(out)


@task(cache=True, resources=Resources(cpu="2", memory="4Gi"))
def featurize(transactions_uri: str, window_days: int) -> str:
    """Aggregate transaction features (cached: re-runs only on new inputs)."""
    data = json.loads(Path(transactions_uri).read_text())
    features = [
        {
            "amount": row["amount"],
            "night": int(row["hour"] < 6),
            "amount_z": row["amount"] / (window_days or 1),
            "label": row["is_fraud"],
        }
        for row in data["rows"]
    ]
    out = _workdir() / "features.json"
    out.write_text(json.dumps(features))
    return str(out)


@task(
    outputs={"model_uri": str, "metrics": dict},
    resources=Resources(cpu="4", memory="8Gi", gpu=1, gpu_type="NVIDIA_TESLA_T4"),
)
def train(features_uri: str, learning_rate: float, epochs: int) -> dict:
    """Fit a (toy) fraud scorer and report training metrics."""
    features = json.loads(Path(features_uri).read_text())
    positives = sum(f["label"] for f in features)
    # Stand-in for a real fit loop.
    weights = {"amount": learning_rate * 3, "night": learning_rate * 7}
    model = {"weights": weights, "epochs": epochs}
    blob = json.dumps(model, sort_keys=True).encode()
    out = _workdir() / f"model_{hashlib.sha256(blob).hexdigest()[:12]}.json"
    out.write_text(json.dumps(model))
    return {
        "model_uri": str(out),
        "metrics": {"train_rows": len(features), "positives": positives, "epochs": epochs},
    }


@task(retry=RetryPolicy(max_attempts=2, delay_seconds=10.0))
def evaluate(model_uri: str, features_uri: str, min_precision: float) -> dict:
    """Score the held-out set and gate on a precision floor."""
    model = json.loads(Path(model_uri).read_text())
    features = json.loads(Path(features_uri).read_text())
    scored = sum(
        1
        for f in features
        if f["amount"] * model["weights"]["amount"] + f["night"] * model["weights"]["night"] > 1
    )
    precision = 0.97  # simulated holdout precision
    if precision < min_precision:
        raise ValueError(f"precision {precision} below floor {min_precision}")
    return {"precision": precision, "flagged": scored, "passed": True}


@pipeline(name="fraud-detection", schedule="0 6 * * *", tags=("ml", "fraud"))
def fraud_pipeline(
    raw_path: str = "s3://warehouse/transactions/",
    sample_rows: int = 500,
    window_days: int = 30,
    learning_rate: float = 0.01,
    epochs: int = 5,
    min_precision: float = 0.9,
):
    """Daily fraud-model training: ingest -> featurize -> train -> evaluate."""
    transactions = ingest(raw_path=raw_path, sample_rows=sample_rows)
    features = featurize(transactions_uri=transactions, window_days=window_days)
    trained = train(features_uri=features, learning_rate=learning_rate, epochs=epochs)
    evaluate(
        model_uri=trained["model_uri"],
        features_uri=features,
        min_precision=min_precision,
    )
