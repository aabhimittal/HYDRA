# Capability matrix — per-cell notes

Generated behavior and rationale for every IR capability on every backend. Issue codes refer to
what `hydraml validate` reports.

## Runtime parameters (`RUNTIME_PARAMS`)

| Backend | Status | Notes |
|---|---|---|
| Airflow | **compile-time bound** (`CAP001`, info) | Airflow's runtime params are stringly-typed Jinja context resolved inside operators; folding IR params at compile time keeps them typed. Bound values are recorded in the generated file's docstring. Override with `--param k=v` and recompile. |
| Prefect | native | IR params become typed flow parameters with defaults. `--param` overrides become the new defaults in the signature. |
| Kubeflow | native | IR params become typed pipeline parameters (protobuf-backed). |
| Local | native | `hydraml run --param k=v`, coerced using the IR type. |

## Cron schedule (`CRON_SCHEDULE`)

| Backend | Status | Notes |
|---|---|---|
| Airflow | embedded | `schedule=` on the `@dag`; `start_date` is fixed (configurable via `--opt start_date=YYYY-MM-DD`) and `catchup=False`. |
| Prefect | embedded, deployment-scoped (`PF001`, info) | Prefect schedules live on deployments, not flows; the generated `__main__` block wires the cron into `flow.serve()`. |
| Kubeflow | **not embeddable** (`CAP002`, warning) | KFP schedules are RecurringRun API objects, separate from the pipeline spec. Create one after uploading. |
| Local | not applicable (`CAP002`, warning) | The local runner is invoked, not scheduled. |

## Retries (`RETRIES`) and backoff (`RETRY_BACKOFF_FACTOR`)

IR semantics: `max_attempts` = total attempts; delay before retry *n* = `delay_seconds * backoff^n`.

| Backend | Status | Notes |
|---|---|---|
| Airflow | retries ✅, factor ✗ (`CAP004`, warning) | `retries=attempts-1`, `retry_delay=timedelta(...)`. Only a doubling boolean exists (`retry_exponential_backoff`); any factor ≠ 1 maps to it. |
| Prefect | exact | The IR schedule is expanded to an explicit `retry_delay_seconds=[...]` list, so *any* factor is represented exactly. |
| Kubeflow | exact | `set_retry(num_retries=..., backoff_duration=..., backoff_factor=...)`. |
| Local | exact | Reference implementation; tests assert the sleep schedule. |

## Task caching (`TASK_CACHING`)

IR semantics: opt-in per task; a cached task re-runs only when its inputs (or code) change.

| Backend | Status | Notes |
|---|---|---|
| Airflow | **unsupported** (`CAP005`, warning) | No memoization primitive exists; the task simply re-runs. |
| Prefect | `cache_policy=INPUTS` | Closest native analogue (input-keyed). Code changes don't invalidate — a known, documented delta. |
| Kubeflow | normalized | KFP caches **by default**, the opposite default from the IR. The backend emits `set_caching_options(True/False)` on every task so opt-in semantics survive. |
| Local | content-addressed | SHA-256 of (fn_ref, function source, resolved inputs) → JSON store in `.hydra_cache/`. Strictest interpretation: code changes do invalidate. |

## Per-task resources (`PER_TASK_RESOURCES`) and GPUs (`GPU_RESOURCES`)

IR semantics: Kubernetes quantity strings (`cpu="2"`, `memory="4Gi"`), integer GPU count plus
accelerator type.

| Backend | Status | Notes |
|---|---|---|
| Airflow | advisory (`AF001`, info) | Emitted as `executor_config.pod_override` with requests+limits; enforced only under KubernetesExecutor. GPUs become `nvidia.com/gpu` entries. |
| Prefect | advisory (`CAP006`/`CAP007` warning, `PF002` info) | Prefect allocates infrastructure per work pool, not per task. Requests are emitted as routing tags (`cpu:2`, `gpu:1`). |
| Kubeflow | enforced | `set_cpu_limit` / `set_memory_limit` / `set_accelerator_type` + `set_accelerator_limit`. A GPU request **requires** an accelerator type (`KF003`, error) — either `Resources(gpu_type=...)` or `--opt default_accelerator=...`. |
| Local | validated only (`CAP006`/`CAP007`, warning) | Declared resources are checked for shape but nothing is enforced in-process. |

## Multi-output tasks (`MULTI_OUTPUT`)

IR semantics: `@task(outputs={"model_uri": str, "metrics": dict})`; the function returns a dict
with those keys; downstream selects `handle["model_uri"]`.

| Backend | Mapping |
|---|---|
| Airflow | `multiple_outputs=True` + XComArg dict indexing |
| Prefect | dict return + plain indexing (sequential flow execution materializes it) |
| Kubeflow | `NamedTuple` component return + `task.outputs["name"]` |
| Local | dict return, contract-checked (missing keys fail *without* retrying) |

## Importable task functions (`CAP009`)

Codegen backends emit `from your.module import fn` — so task functions must be importable
top-level names. Functions defined inside other functions (`<locals>` in the qualname) or in
`__main__` are compile errors for Airflow/Prefect/Kubeflow. The local backend has no such
restriction (it resolves through a decoration-time registry), which keeps notebook/REPL iteration
friction-free.
