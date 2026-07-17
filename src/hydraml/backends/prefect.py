"""Prefect backend: compiles the IR to a Prefect 3 flow file.

Notable semantic mappings (all surfaced as issues):

- **Runtime params are native** - flow parameters keep IR types and defaults.
- **Retry backoff is exact**: the IR's delay/backoff schedule is expanded to
  an explicit ``retry_delay_seconds`` list, so any factor is representable.
- **Caching maps to ``cache_policy=INPUTS``** (re-run only when inputs
  change) - the closest analogue of HYDRA's content-addressed caching.
- **Resources are advisory**: Prefect allocates infrastructure per work pool,
  not per task; requests are emitted as task tags (``cpu:2``) for routing.
- **Schedules attach to deployments**, not flows; the generated file wires
  the cron into ``flow.serve()`` under ``__main__``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ir.graph import topological_order
from ..ir.model import OutputRef, Pipeline, TaskSpec
from ..issues import Issue, Severity
from ._codegen import Emitter, header, import_aliases, render_input, result_var, sanitize
from .base import Backend, Capability, register

_PY_TYPES = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "dict": "dict",
    "list": "list",
    "Any": "object",
}


@register
class PrefectBackend(Backend):
    name = "prefect"
    capabilities = frozenset(
        {
            Capability.RUNTIME_PARAMS,
            Capability.CRON_SCHEDULE,
            Capability.RETRIES,
            Capability.RETRY_BACKOFF_FACTOR,
            Capability.TASK_CACHING,
            Capability.MULTI_OUTPUT,
        }
    )

    def extra_issues(self, pipeline: Pipeline) -> list[Issue]:
        issues = []
        if pipeline.schedule:
            issues.append(
                Issue(
                    Severity.INFO,
                    "PF001",
                    "Prefect schedules attach to deployments; the cron is wired"
                    " into flow.serve() in the generated __main__ block",
                )
            )
        for tname, tspec in pipeline.tasks.items():
            if not tspec.resources.is_empty:
                issues.append(
                    Issue(
                        Severity.INFO,
                        "PF002",
                        "resources are emitted as task tags (e.g. 'cpu:2') for"
                        " work-pool routing; enforcement happens at the work"
                        " pool, not the task",
                        task=tname,
                    )
                )
        return issues

    def emit(self, pipeline: Pipeline, outdir: Path, params: dict[str, Any]) -> list[Path]:
        flow_name = sanitize(pipeline.name)
        emitter = Emitter()
        emitter.lines(header(pipeline, self.name))
        emitter.blank()
        emitter.line("from __future__ import annotations")
        emitter.blank()
        emitter.line("from prefect import flow, task")
        if any(t.cache for t in pipeline.tasks.values()):
            emitter.line("from prefect.cache_policies import INPUTS")
        emitter.blank()
        aliases = import_aliases(pipeline)
        for module, attr, alias in aliases.values():
            emitter.line(f"from {module} import {attr} as {alias}")
        emitter.blank(2)

        # Task wrappers are bound once per distinct IR task (invocations of the
        # same @task can carry the same metadata, so per-task binding is exact).
        for tname in pipeline.tasks:
            self._emit_task_binding(emitter, pipeline.tasks[tname], aliases)
            emitter.blank()
        emitter.blank()

        signature = ", ".join(
            f"{p.name}: {_PY_TYPES[p.type]}" + ("" if p.required else f" = {params[p.name]!r}")
            for p in pipeline.params.values()
        )
        flow_args = [f"name={pipeline.name!r}"]
        if pipeline.description:
            flow_args.append(f"description={pipeline.description!r}")
        emitter.line(f"@flow({', '.join(flow_args)})")
        emitter.line(f"def {flow_name}({signature}):")
        emitter.indent()
        for tname in topological_order(pipeline):
            self._emit_task_call(emitter, pipeline, pipeline.tasks[tname], params)
        last = topological_order(pipeline)[-1]
        emitter.line(f"return {result_var(last)}")
        emitter.dedent()
        emitter.blank(2)

        emitter.line('if __name__ == "__main__":')
        emitter.indent()
        if pipeline.schedule:
            emitter.line(f"{flow_name}.serve(name={pipeline.name!r}, cron={pipeline.schedule!r})")
        else:
            emitter.line(f"{flow_name}()")
        emitter.dedent()

        path = outdir / f"{flow_name}_prefect.py"
        path.write_text(emitter.source())
        return [path]

    def _emit_task_binding(
        self, emitter: Emitter, tspec: TaskSpec, aliases: dict[str, tuple[str, str, str]]
    ) -> None:
        alias = aliases[tspec.fn_ref][2]
        args: list[str] = [f"name={tspec.name!r}"]
        if tspec.retry.retries > 0:
            args.append(f"retries={tspec.retry.retries}")
            delays = tspec.retry.delays()
            if len(set(delays)) == 1:
                args.append(f"retry_delay_seconds={delays[0]!r}")
            else:
                args.append(f"retry_delay_seconds={delays!r}")
        if tspec.cache:
            args.append("cache_policy=INPUTS")
        tags = self._tags(tspec)
        if tags:
            args.append(f"tags={tags!r}")
        emitter.line(f"{sanitize(tspec.name)} = task({', '.join(args)})({alias}.fn)")

    def _emit_task_call(
        self, emitter: Emitter, pipeline: Pipeline, tspec: TaskSpec, params: dict[str, Any]
    ) -> None:
        def output_expr(ref: OutputRef, upstream: TaskSpec) -> str:
            var = result_var(upstream.name)
            # Multi-output tasks return a dict; sequential flow execution means
            # the value is already materialized here.
            return var if upstream.single_output else f"{var}[{ref.output!r}]"

        rendered = {
            arg: render_input(
                value, params, pipeline.tasks, runtime_params=True, output_expr=output_expr
            )
            for arg, value in sorted(tspec.inputs.items())
        }
        call_args = ", ".join(f"{arg}={expr}" for arg, expr in rendered.items())
        emitter.line(f"{result_var(tspec.name)} = {sanitize(tspec.name)}({call_args})")

    @staticmethod
    def _tags(tspec: TaskSpec) -> list[str]:
        tags = []
        if tspec.resources.cpu:
            tags.append(f"cpu:{tspec.resources.cpu}")
        if tspec.resources.memory:
            tags.append(f"memory:{tspec.resources.memory}")
        if tspec.resources.gpu:
            tags.append(f"gpu:{tspec.resources.gpu}")
        return tags
