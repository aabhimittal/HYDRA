"""Airflow backend: compiles the IR to a TaskFlow-style DAG file.

Targets Airflow >= 2.7. Notable semantic mappings (all surfaced as issues):

- **Params are compile-time bound.** Airflow's native runtime params are
  stringly-typed Jinja context; binding at compile time preserves the IR's
  types at the cost of needing a recompile to change values.
- **Retry backoff is boolean.** Airflow exposes ``retry_exponential_backoff``
  (doubling) rather than an arbitrary factor; any factor != 1 maps to True.
- **No task caching.** Airflow has no memoization primitive; ``cache=True``
  degrades to a warning.
- **Resources are advisory.** Emitted as ``executor_config`` in the
  KubernetesExecutor's ``pod_override`` shape; other executors ignore it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ir.graph import topological_order
from ..ir.model import OutputRef, Pipeline, TaskSpec
from ..issues import Issue, Severity
from ._codegen import Emitter, header, import_aliases, render_input, result_var, sanitize
from .base import Backend, Capability, register


@register
class AirflowBackend(Backend):
    name = "airflow"
    capabilities = frozenset(
        {
            Capability.CRON_SCHEDULE,
            Capability.RETRIES,
            Capability.PER_TASK_RESOURCES,
            Capability.GPU_RESOURCES,
            Capability.MULTI_OUTPUT,
        }
    )

    @classmethod
    def option_names(cls) -> tuple[str, ...]:
        return ("start_date",)  # "YYYY-MM-DD", default 2024-01-01

    def extra_issues(self, pipeline: Pipeline) -> list[Issue]:
        issues = []
        for tname, tspec in pipeline.tasks.items():
            if not tspec.resources.is_empty:
                issues.append(
                    Issue(
                        Severity.INFO,
                        "AF001",
                        "resources are emitted as KubernetesExecutor"
                        " executor_config (pod_override); other executors"
                        " ignore them",
                        task=tname,
                    )
                )
        return issues

    def emit(self, pipeline: Pipeline, outdir: Path, params: dict[str, Any]) -> list[Path]:
        dag_id = sanitize(pipeline.name)
        emitter = Emitter()
        emitter.lines(
            header(
                pipeline,
                self.name,
                extra=["Parameters were bound at compile time:"]
                + [f"    {k} = {v!r}" for k, v in params.items()],
            )
        )
        emitter.blank()
        emitter.line("from __future__ import annotations")
        emitter.blank()
        emitter.line("from datetime import datetime, timedelta")
        emitter.blank()
        emitter.line("from airflow.decorators import dag, task")
        emitter.blank()
        aliases = import_aliases(pipeline)
        for module, attr, alias in aliases.values():
            emitter.line(f"from {module} import {attr} as {alias}")
        emitter.blank(2)

        start = self.options.get("start_date", "2024-01-01")
        year, month, day = (int(part) for part in str(start).split("-"))
        emitter.line("@dag(")
        emitter.indent()
        emitter.line(f"dag_id={dag_id!r},")
        if pipeline.description:
            emitter.line(f"description={pipeline.description!r},")
        emitter.line(f"schedule={pipeline.schedule!r},")
        emitter.line(f"start_date=datetime({year}, {month}, {day}),")
        emitter.line("catchup=False,")
        emitter.line(f"tags={sorted({*pipeline.tags, 'hydra'})!r},")
        emitter.dedent()
        emitter.line(")")
        emitter.line(f"def {dag_id}():")
        emitter.indent()

        order = topological_order(pipeline)
        for tname in order:
            self._emit_task_def(emitter, pipeline.tasks[tname], aliases)
            emitter.blank()
        for tname in order:
            self._emit_task_call(emitter, pipeline, pipeline.tasks[tname], params)
        emitter.dedent()
        emitter.blank(2)
        emitter.line(f"{dag_id}()")

        path = outdir / f"{dag_id}_airflow.py"
        path.write_text(emitter.source())
        return [path]

    def _emit_task_def(
        self, emitter: Emitter, tspec: TaskSpec, aliases: dict[str, tuple[str, str, str]]
    ) -> None:
        fn_name = sanitize(tspec.name)
        alias = aliases[tspec.fn_ref][2]
        args: list[str] = [f"task_id={tspec.name!r}"]
        if tspec.retry.retries > 0:
            args.append(f"retries={tspec.retry.retries}")
            args.append(f"retry_delay=timedelta(seconds={tspec.retry.delay_seconds!r})")
            if tspec.retry.backoff != 1.0:
                # Lossy: Airflow only supports doubling (see CAP004).
                args.append("retry_exponential_backoff=True")
        if not tspec.single_output:
            args.append("multiple_outputs=True")
        if not tspec.resources.is_empty:
            args.append(f"executor_config={self._executor_config(tspec)!r}")
        params = sorted(tspec.inputs)
        emitter.line(f"@task({', '.join(args)})")
        emitter.line(f"def {fn_name}({', '.join(params)}):")
        emitter.indent()
        call = f"{alias}.fn({', '.join(f'{p}={p}' for p in params)})"
        if tspec.single_output:
            emitter.line(f"return {call}")
        else:
            emitter.line(f"result = {call}")
            emitter.line(f"return {{key: result[key] for key in {list(tspec.outputs)!r}}}")
        emitter.dedent()

    def _emit_task_call(
        self, emitter: Emitter, pipeline: Pipeline, tspec: TaskSpec, params: dict[str, Any]
    ) -> None:
        def output_expr(ref: OutputRef, upstream: TaskSpec) -> str:
            var = result_var(upstream.name)
            return var if upstream.single_output else f"{var}[{ref.output!r}]"

        rendered = {
            arg: render_input(
                value, params, pipeline.tasks, runtime_params=False, output_expr=output_expr
            )
            for arg, value in sorted(tspec.inputs.items())
        }
        call_args = ", ".join(f"{arg}={expr}" for arg, expr in rendered.items())
        emitter.line(f"{result_var(tspec.name)} = {sanitize(tspec.name)}({call_args})")

    @staticmethod
    def _executor_config(tspec: TaskSpec) -> dict[str, Any]:
        requests: dict[str, Any] = {}
        if tspec.resources.cpu:
            requests["cpu"] = tspec.resources.cpu
        if tspec.resources.memory:
            requests["memory"] = tspec.resources.memory
        if tspec.resources.gpu:
            requests["nvidia.com/gpu"] = str(tspec.resources.gpu)
        return {
            "pod_override": {
                "spec": {
                    "containers": [
                        {
                            "name": "base",
                            "resources": {"requests": requests, "limits": dict(requests)},
                        }
                    ]
                }
            }
        }
