# HYDRA architecture

This document explains how HYDRA is put together and, more importantly, *why* each piece is shaped
the way it is. The short version: HYDRA is a compiler. Everything follows from taking that framing
seriously.

## Compilation pipeline

```
 spec string ──► loader ──► tracing DSL ──► IR ──► validation ──► codegen ──► native artifact
 (module/file/yaml)         (@task/@pipeline)      (structural +   (backend-
                                                    capability)     specific)
```

1. **Load** (`hydraml.loader`). A spec string (`pkg.module:attr`, `path/file.py`, `spec.yaml`)
   resolves to a `Pipeline` IR object. File loads register the module in `sys.modules` under its
   CWD-relative dotted name so that function references recorded during tracing stay importable —
   which is what lets generated code emit real import statements.
2. **Trace** (`hydraml.dsl`). `@pipeline` executes the decorated function once, at definition
   time, with `ParamHandle` placeholders instead of arguments. Task calls don't run; they record
   `TaskSpec` nodes and return symbolic handles. The decorator evaluates to the finished
   `Pipeline` — there is no separate "build" step to forget.
3. **Validate** (`hydraml.ir.graph` + `hydraml.backends.base`). Structural checks (dangling
   references, cycles, duplicate outputs, non-JSON literals) are backend-independent. Capability
   checks compare the pipeline's demands against the target backend's declared `Capability` set.
4. **Compile** (`hydraml.backends.*`). Each backend emits deterministic, idiomatic source for its
   target. Validation errors abort before any file is written.

## The IR

A `Pipeline` is:

| Field | Meaning |
|---|---|
| `params` | typed pipeline inputs (`str/int/float/bool/dict/list/Any`), required or defaulted |
| `tasks` | ordered DAG nodes; each has `fn_ref`, `inputs`, `outputs`, `retry`, `resources`, `cache` |
| `schedule` | a cron expression, or `None` |

Task inputs are one of three reference types — `ParamRef`, `OutputRef(task, output)`, or
`LiteralValue` — never live Python objects. Task bodies are referenced by import path
(`module:qualname`), never embedded. Both choices exist to keep the IR serializable: the YAML form
(`hydraml inspect`) round-trips losslessly and is the stable interface for tooling.

### Why the IR is small

Every field in the IR must be *compilable to all backends or honestly refusable by some*. That
discipline is the difference between a meta-layer and a lowest-common-denominator wrapper that
quietly drops semantics. Features enter the IR only with a defined mapping (or a defined
degradation story) for every backend — which is why dynamic fan-out and conditionals are on the
roadmap rather than half-supported today.

### Data passing: references, not blobs

Values that cross task boundaries must be JSON-serializable and small — parameters, URIs, metric
dicts. The DSL rejects non-serializable literals at trace time; the local runner's JSON cache
enforces the same contract at run time.

This isn't a limitation adopted for convenience. It is the *only* data-passing model with a clean
mapping onto all three backends at once:

| Backend | Mechanism | Constraint |
|---|---|---|
| Airflow | XCom (TaskFlow return values) | metadata-DB row; must be small & serializable |
| Prefect | task return values | held in flow-run memory; results storage serializes |
| Kubeflow | component parameters | protobuf-encoded; artifacts are a separate typed system |

Heavy data (datasets, weights) travels by reference: a task writes to object storage and returns
the URI. This also makes lineage explicit in the graph.

## Capability negotiation

Each backend declares a `frozenset[Capability]`:

```python
class Capability(str, Enum):
    RUNTIME_PARAMS, CRON_SCHEDULE, RETRIES, RETRY_BACKOFF_FACTOR,
    TASK_CACHING, PER_TASK_RESOURCES, GPU_RESOURCES, MULTI_OUTPUT
```

`Backend.validate()` walks the pipeline's demands and emits `Issue`s with three severities:

- **ERROR** — cannot compile (e.g. KFP GPU request without an accelerator type; a task function
  defined inside another function, which cannot be imported by generated code).
- **WARNING** — compiles, but the feature degrades (Airflow drops caching; Prefect can't enforce
  resources).
- **INFO** — compiles and works, but the *meaning* shifts (Airflow params bind at compile time;
  KFP resources are limits, not requests).

The severity taxonomy is the design: a migration between orchestrators becomes a reviewable list
of semantic differences instead of a debugging campaign.

### Semantics HYDRA actively normalizes

- **Caching defaults.** KFP caches by default; Airflow/Prefect don't. The IR says caching is
  opt-in per task, so the KFP backend emits `set_caching_options(...)` on *every* task.
- **Retry counting.** IR `max_attempts` counts total attempts; all backends express retries as
  attempts-1, and the conversion lives in one place (`RetryPolicy.retries`).
- **Backoff.** IR = `delay * backoff^n`. Prefect gets the exact schedule as an explicit delay
  list; KFP gets `backoff_factor`; Airflow only has a doubling boolean, so any factor ≠ 1 maps to
  `retry_exponential_backoff=True` **and** a `CAP004` warning.

## Why codegen (and not a runtime adapter)

The obvious alternative is a runtime shim: HYDRA-the-library sits between the orchestrator and the
task, translating calls. That was rejected deliberately:

1. **Debuggability.** With codegen, a failing Airflow task is a plain TaskFlow task; stack traces
   contain the user's function directly. A shim inserts itself into every frame.
2. **Operational trust.** Platform teams can review the generated DAG/flow/pipeline as ordinary
   code. Nothing opaque runs on the scheduler.
3. **No dependency coupling.** `hydraml` imports none of the three orchestrators; version churn in
   Airflow/Prefect/KFP affects the (regenerable) output, not the framework.
4. **A clean exit.** If a team abandons HYDRA, the compiled artifacts keep working as-is.

The cost: generated code must be deterministic and treated as an artifact. Golden-file tests pin
the exact output; codegen changes show up as reviewed golden diffs.

### Backend-specific mappings worth knowing

**Airflow (TaskFlow, ≥ 2.7).** Each IR task becomes an inner `@task` function that calls the
imported user function; dependencies flow through XComArgs. Multi-output tasks use
`multiple_outputs=True` and dict indexing. Resources emit as `executor_config.pod_override`
(KubernetesExecutor shape). Params are folded to compile-time constants — the bound values are
recorded in the generated file's docstring.

**Prefect (≥ 3).** Task metadata binds via `task(...)(fn)` at module level; the flow keeps IR
params as real typed flow parameters. Caching maps to `cache_policy=INPUTS`. The cron schedule
attaches where Prefect wants it: `flow.serve(cron=...)` in the `__main__` block.

**Kubeflow (KFP v2).** Each distinct task function becomes a `@dsl.component` whose body imports
the user function *inside* the component (KFP serializes component source, so top-level imports
would be lost). Consequence: the user's package must exist in the component image —
`--opt base_image=...` / `--opt packages=...` are how you say where. Multi-output tasks return
`NamedTuple`s; input types are inferred from what flows into them (param types, upstream output
types, literal types), and untyped values degrade to `str` with a warning.

## The local backend

`hydraml run` executes the IR in-process: topological order, full retry schedule (injectable
sleeper for tests), content-addressed caching (SHA-256 of function source + resolved inputs, JSON
value store), downstream skipping on failure. It is the *reference semantics* for the IR — tests
assert behavior against it — and it deliberately shares the validation path with the codegen
backends, so `run` catches the same structural errors `compile` would.

It resolves callables through a decoration-time registry first and importlib second, so pipelines
defined in scripts and REPLs run locally even though codegen backends would reject them
(`CAP009`) for not being importable.

## Known limitations

- No dynamic fan-out or conditionals in the IR yet (roadmap; each needs a defined mapping for all
  three backends before it can enter the IR).
- One pipeline → one file; no cross-pipeline composition.
- `hydraml submit` (deploying compiled artifacts to live backends) is not built; compile output is
  handed to each system's own deployment flow.
- Airflow resources assume KubernetesExecutor for enforcement; other executors treat them as
  documentation.
