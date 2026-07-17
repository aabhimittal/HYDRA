# Examples

## fraud_detection

A daily fraud-model training pipeline: `ingest → featurize → train → evaluate`, with a fan-in
(evaluate consumes both the trained model and the feature set). Task bodies are stdlib-only
simulations so the example runs anywhere; the orchestration structure is the point.

It exercises every IR feature: typed params with defaults, retries with backoff, opt-in caching,
CPU/memory/GPU resources, multi-output tasks, and a cron schedule.

```bash
# execute in-process (watch the cache kick in on the second run)
hydraml run examples/fraud_detection/pipeline.py
hydraml run examples/fraud_detection/pipeline.py --param epochs=20

# what does each backend lose?
hydraml validate examples/fraud_detection/pipeline.py

# compile to all three orchestrators
hydraml compile examples/fraud_detection/pipeline.py --backend airflow  -o build/
hydraml compile examples/fraud_detection/pipeline.py --backend prefect  -o build/
hydraml compile examples/fraud_detection/pipeline.py --backend kubeflow -o build/ \
    --opt base_image=python:3.11 --opt packages=hydraml
```

The compiled outputs for this pipeline (with default options) are checked in as the codegen
goldens under [`tests/golden/`](../tests/golden) — read them to see exactly what HYDRA emits for
each backend.
