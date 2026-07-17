# HYDRA

**One pipeline definition, many orchestrators.** HYDRA is a poly-orchestration *compiler* for ML
pipelines: you author a pipeline once against a small, backend-neutral intermediate representation
(IR), and HYDRA validates it against — and compiles it to — **Kubeflow Pipelines**, **Airflow**, or
**Prefect**. A built-in local backend executes the same IR in-process for fast iteration.

```
                          ┌──────────────────────┐
   @task / @pipeline ───► │      HYDRA IR        │ ───►  validate (capability
   (authoring DSL)        │  tasks · params ·    │        negotiation per backend)
                          │  retries · resources │
                          │  caching · schedule  │ ───►  compile (native codegen)
                          └──────────┬───────────┘
                                     │
             ┌───────────────┬───────┴───────┬───────────────┐
             ▼               ▼               ▼               ▼
        Airflow DAG     Prefect flow    KFP v2 pipeline   local runner
        (TaskFlow)      (Prefect 3)     (components)      (in-process)
```

## Why this exists

Kubeflow, Airflow, and Prefect are not interchangeable — they disagree about *when parameters
bind*, *whether tasks cache*, *how retries back off*, and *who owns compute resources*. Teams that
migrate between them (or run more than one) end up rewriting pipelines by hand and silently losing
semantics along the way.

HYDRA's position: treat the orchestrators as **execution backends**, the way a compiler treats
CPU architectures.

1. **A deliberately small IR.** A pipeline is a DAG of tasks with typed parameters, retry
   policies, resource requests, opt-in caching, and a cron schedule. Nothing backend-specific
   leaks in.
2. **Capability negotiation instead of silent loss.** Every backend declares what it can honor.
   `hydraml validate` reports exactly what degrades (warning), what changes meaning (info), and
   what cannot compile (error) — *before* anything runs. Feature gaps are data, not surprises.
3. **Code generation, not runtime wrapping.** `hydraml compile` emits idiomatic, reviewable
   source in each tool's native dialect (TaskFlow API, Prefect 3 flows, KFP v2 components). The
   artifact is plain code you can diff, commit, and deploy — HYDRA is not in the runtime loop,
   and `hydraml` itself depends on none of the three orchestrators.

## Quickstart

```bash
pip install -e .            # zero orchestrator dependencies required
```

Author once:

```python
from hydraml import Resources, RetryPolicy, pipeline, task

@task(retry=RetryPolicy(max_attempts=3, delay_seconds=5.0, backoff=2.0))
def ingest(raw_path: str, sample_rows: int) -> str: ...

@task(cache=True, resources=Resources(cpu="2", memory="4Gi"))
def featurize(transactions_uri: str, window_days: int) -> str: ...

@task(outputs={"model_uri": str, "metrics": dict},
      resources=Resources(cpu="4", memory="8Gi", gpu=1, gpu_type="NVIDIA_TESLA_T4"))
def train(features_uri: str, learning_rate: float, epochs: int) -> dict: ...

@pipeline(name="fraud-detection", schedule="0 6 * * *")
def fraud_pipeline(raw_path: str = "s3://warehouse/transactions/", ...):
    transactions = ingest(raw_path=raw_path, sample_rows=sample_rows)
    features = featurize(transactions_uri=transactions, window_days=window_days)
    trained = train(features_uri=features, learning_rate=learning_rate, epochs=epochs)
    evaluate(model_uri=trained["model_uri"], features_uri=features, min_precision=min_precision)
```

Run anywhere:

```bash
# iterate locally: retries, caching, and data-passing semantics all honored
hydraml run examples/fraud_detection/pipeline.py

# see what each backend can and cannot honor
hydraml validate examples/fraud_detection/pipeline.py

# compile to native code
hydraml compile examples/fraud_detection/pipeline.py --backend airflow  -o build/
hydraml compile examples/fraud_detection/pipeline.py --backend prefect  -o build/
hydraml compile examples/fraud_detection/pipeline.py --backend kubeflow -o build/ \
    --opt base_image=ghcr.io/acme/train:1

# inspect the IR / export the graph
hydraml inspect examples/fraud_detection/pipeline.py
hydraml graph examples/fraud_detection/pipeline.py --format mermaid
hydraml backends
```

Sample `validate` output — the capability gap is explicit, per task:

```
== airflow ==
INFO    CAP001: airflow: parameters are bound at compile time; pass --param to override, then recompile
WARNING CAP004: airflow: backoff factor 2.0 is approximated (backend supports only on/off exponential backoff) [task: ingest]
WARNING CAP005: airflow: task caching is not supported; the task will re-run on every pipeline run [task: featurize]

== kubeflow ==
WARNING CAP002: kubeflow: cron schedule '0 6 * * *' cannot be embedded in the compiled definition; create a RecurringRun via the KFP API/UI
```

## Capability matrix

| Capability | Airflow | Prefect | Kubeflow | Local |
|---|---|---|---|---|
| Runtime parameters | compile-time bound¹ | ✅ | ✅ | ✅ |
| Cron schedule in artifact | ✅ | via `flow.serve()` | ✗ (RecurringRun API) | ✗ |
| Retries | ✅ | ✅ | ✅ | ✅ |
| Arbitrary backoff factor | ✗ (on/off only) | ✅ (explicit delay list) | ✅ (`set_retry`) | ✅ |
| Task caching | ✗ | ✅ (`cache_policy=INPUTS`) | ✅ (normalized²) | ✅ (content hash) |
| Per-task resources | advisory (`executor_config`) | advisory (tags) | ✅ | validated only |
| GPUs | advisory | ✗ | ✅ (accelerator type required) | ✗ |
| Multi-output tasks | ✅ (XCom dict) | ✅ | ✅ (NamedTuple) | ✅ |

¹ Airflow's native runtime params are stringly-typed Jinja context; HYDRA binds params at compile
time to preserve IR types, and says so via `CAP001`.
² KFP caches *by default*; HYDRA emits `set_caching_options(...)` on every task so the IR's opt-in
semantics survive. Semantic normalization like this is most of the point.

Full per-cell notes: [docs/capability-matrix.md](docs/capability-matrix.md).

## Design tenets

- **Pass references, not blobs.** Values crossing task boundaries must be small and
  JSON-serializable — parameters, URIs, metric dicts. Heavy artifacts live in object storage and
  travel by reference. This is the only data-passing model that maps cleanly onto XCom, Prefect
  return values, and KFP component parameters simultaneously, and the DSL enforces it at trace
  time.
- **The IR is the product.** It serializes to YAML (`hydraml inspect`), diffs in code review, and
  round-trips losslessly — `schema_version` is there so it can evolve.
- **Generated code is a first-class artifact.** Deterministic output, golden-file tested; a codegen
  change is a reviewed diff to the goldens.
- **Fail loudly at the boundary.** Anything that can't be traced (f-strings on task outputs,
  non-serializable literals, `*args` signatures, branching on a future value) raises immediately
  at definition time with an explanation, not at 3 a.m. on the scheduler.

## What HYDRA is *not*

- Not a fourth orchestrator: no scheduler, no runtime services, no agents.
- Not an abstraction that hides the backends: the compiled artifact is idiomatic code for the
  target, and every semantic difference is reported, not papered over.
- Not (yet) a superset of every backend feature — see the roadmap.

## Repository layout

```
src/hydraml/
  ir/          # the IR: model, graph algorithms, YAML serde
  dsl/         # @task / @pipeline tracing, symbolic handles
  backends/    # base (capability negotiation) + airflow / prefect / kubeflow / local
  loader.py    # module / file / YAML spec resolution
  cli.py       # backends · inspect · graph · validate · compile · run
examples/      # runnable fraud-detection pipeline (stdlib-only task bodies)
tests/         # unit + CLI tests; tests/golden/ pins the codegen output
docs/          # architecture deep-dive, capability matrix notes
```

## Roadmap

- Dynamic fan-out (`map`) — Prefect `.map`, Airflow dynamic task mapping, KFP `ParallelFor`
- Conditional branches — KFP `dsl.If`, Airflow branch operators, plain `if` in Prefect
- Typed artifact I/O (dataset/model references with lineage metadata)
- `hydraml submit` — hand the compiled artifact to a live backend and stream status
- Dagster backend, as a proof that the IR stays backend-count-agnostic

## FAQ

**Why is the package `hydraml` and not `hydra`?** The `hydra` import namespace belongs to
[hydra-core](https://hydra.cc) (Facebook's config framework), which is ubiquitous in ML
environments, and the `hydra` binary name belongs to a well-known network tool on most Linux
distros. Branding is HYDRA; imports are boring on purpose.

**Why codegen instead of a runtime adapter?** A runtime adapter puts HYDRA between the scheduler
and your tasks forever — every backend upgrade, every debugging session goes through the shim.
Generated native code has no such coupling: ops teams review and deploy exactly what they'd have
written by hand, and deleting HYDRA leaves you with three working codebases instead of zero.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

MIT licensed.
