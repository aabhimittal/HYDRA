"""Kubeflow backend: compiles the IR to a KFP v2 pipeline file.

Each IR task becomes a ``@dsl.component`` whose body imports the user's task
function *inside* the component (KFP serializes component source, so top-level
imports would be lost). This means the task's package must be installed in the
component image - set it via ``--opt base_image=...`` and/or
``--opt packages=pkg1,pkg2``.

Notable semantic mappings (all surfaced as issues):

- **Retries are exact**: ``set_retry`` supports count, delay, and an arbitrary
  backoff factor.
- **Caching defaults differ**: KFP caches *by default*; HYDRA normalizes by
  emitting ``set_caching_options(...)`` explicitly on every task so IR
  semantics (opt-in caching) are preserved.
- **Resources and GPUs are first-class** (``set_cpu_limit`` etc.); a GPU
  request requires an accelerator type (``gpu_type`` or
  ``--opt default_accelerator=...``).
- **Cron schedules are out-of-band**: recurring runs are API objects, not part
  of the pipeline spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ir.graph import topological_order
from ..ir.model import LiteralValue, OutputRef, ParamRef, Pipeline, TaskSpec
from ..issues import Issue, Severity
from ._codegen import Emitter, header, import_aliases, render_input, sanitize
from .base import Backend, Capability, register

_KFP_TYPES = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "dict": "dict",
    "list": "list",
    # KFP v2 requires concrete annotations on component signatures; untyped IR
    # values degrade to str (surfaced as KF002).
    "Any": "str",
}

_DEFAULT_IMAGE = "python:3.11"


@register
class KubeflowBackend(Backend):
    name = "kubeflow"
    capabilities = frozenset(
        {
            Capability.RUNTIME_PARAMS,
            Capability.RETRIES,
            Capability.RETRY_BACKOFF_FACTOR,
            Capability.TASK_CACHING,
            Capability.PER_TASK_RESOURCES,
            Capability.GPU_RESOURCES,
            Capability.MULTI_OUTPUT,
        }
    )

    @classmethod
    def option_names(cls) -> tuple[str, ...]:
        return ("base_image", "packages", "default_accelerator")

    def schedule_hint(self) -> str:
        return (
            "create a RecurringRun via the KFP API/UI after uploading the"
            " compiled pipeline"
        )

    def extra_issues(self, pipeline: Pipeline) -> list[Issue]:
        issues = [
            Issue(
                Severity.INFO,
                "KF001",
                f"components run in image {self.options.get('base_image', _DEFAULT_IMAGE)!r};"
                " the module defining your task functions must be importable"
                " inside it (see --opt base_image / --opt packages)",
            )
        ]
        for tname, tspec in pipeline.tasks.items():
            untyped = [
                arg
                for arg in tspec.inputs
                if _input_type(pipeline, tspec, arg) == "Any"
            ]
            if untyped or any(
                tspec.output_types.get(o, "Any") == "Any" for o in tspec.outputs
            ):
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "KF002",
                        "untyped inputs/outputs degrade to `str` in the KFP"
                        " component signature; add annotations to the task"
                        " function for exact types",
                        task=tname,
                    )
                )
            if tspec.resources.gpu > 0 and not self._accelerator(tspec):
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "KF003",
                        "GPU request needs an accelerator type: set"
                        " Resources(gpu_type=...) or --opt default_accelerator=...",
                        task=tname,
                    )
                )
        return issues

    def emit(self, pipeline: Pipeline, outdir: Path, params: dict[str, Any]) -> list[Path]:
        pipe_fn = sanitize(pipeline.name)
        emitter = Emitter()
        emitter.lines(header(pipeline, self.name))
        emitter.blank()
        emitter.line("from __future__ import annotations")
        emitter.blank()
        if any(not t.single_output for t in pipeline.tasks.values()):
            emitter.line("from typing import NamedTuple")
            emitter.blank()
        emitter.line("from kfp import dsl")
        emitter.blank(2)

        aliases = import_aliases(pipeline)
        # One component per distinct task function (re-invocations share it).
        emitted: set[str] = set()
        for tspec in pipeline.tasks.values():
            if tspec.fn_ref in emitted:
                continue
            emitted.add(tspec.fn_ref)
            self._emit_component(emitter, pipeline, tspec, aliases)
            emitter.blank(2)

        signature = ", ".join(
            f"{p.name}: {_KFP_TYPES[p.type]}" + ("" if p.required else f" = {params[p.name]!r}")
            for p in pipeline.params.values()
        )
        pipe_args = [f"name={pipeline.name!r}"]
        if pipeline.description:
            pipe_args.append(f"description={pipeline.description!r}")
        emitter.line(f"@dsl.pipeline({', '.join(pipe_args)})")
        emitter.line(f"def {pipe_fn}({signature}):")
        emitter.indent()
        for tname in topological_order(pipeline):
            self._emit_task_call(emitter, pipeline, pipeline.tasks[tname], params, aliases)
        emitter.dedent()
        emitter.blank(2)

        emitter.line('if __name__ == "__main__":')
        emitter.indent()
        emitter.line("from kfp import compiler")
        emitter.blank()
        emitter.line(
            f"compiler.Compiler().compile({pipe_fn}, package_path={pipe_fn + '.yaml'!r})"
        )
        emitter.dedent()

        path = outdir / f"{pipe_fn}_kubeflow.py"
        path.write_text(emitter.source())
        return [path]

    # --- components ------------------------------------------------------------

    def _component_name(self, tspec: TaskSpec, aliases: dict) -> str:
        return f"{sanitize(aliases[tspec.fn_ref][1])}_component"

    def _emit_component(
        self, emitter: Emitter, pipeline: Pipeline, tspec: TaskSpec, aliases: dict
    ) -> None:
        module, attr, _ = aliases[tspec.fn_ref]
        comp_args = [f"base_image={self.options.get('base_image', _DEFAULT_IMAGE)!r}"]
        packages = self.options.get("packages")
        if packages:
            pkg_list = [p.strip() for p in str(packages).split(",") if p.strip()]
            comp_args.append(f"packages_to_install={pkg_list!r}")
        params = ", ".join(
            f"{arg}: {_KFP_TYPES[_input_type(pipeline, tspec, arg)]}"
            for arg in sorted(tspec.inputs)
        )
        if tspec.single_output:
            ret = _KFP_TYPES[tspec.output_types.get(tspec.outputs[0], "Any")]
        else:
            fields = ", ".join(
                f'("{o}", {_KFP_TYPES[tspec.output_types.get(o, "Any")]})' for o in tspec.outputs
            )
            ret = f'NamedTuple("Outputs", [{fields}])'
        emitter.line(f"@dsl.component({', '.join(comp_args)})")
        emitter.line(f"def {self._component_name(tspec, aliases)}({params}) -> {ret}:")
        emitter.indent()
        emitter.line(f"from {module} import {attr} as _task_def")
        emitter.blank()
        call = f"_task_def.fn({', '.join(f'{a}={a}' for a in sorted(tspec.inputs))})"
        if tspec.single_output:
            emitter.line(f"return {call}")
        else:
            emitter.line(f"result = {call}")
            emitter.line("from collections import namedtuple")
            emitter.blank()
            emitter.line(f'outputs = namedtuple("Outputs", {list(tspec.outputs)!r})')
            emitter.line(
                "return outputs(" + ", ".join(f"{o}=result[{o!r}]" for o in tspec.outputs) + ")"
            )
        emitter.dedent()

    # --- pipeline body -----------------------------------------------------------

    def _emit_task_call(
        self,
        emitter: Emitter,
        pipeline: Pipeline,
        tspec: TaskSpec,
        params: dict[str, Any],
        aliases: dict,
    ) -> None:
        def output_expr(ref: OutputRef, upstream: TaskSpec) -> str:
            var = f"{sanitize(upstream.name)}_task"
            if upstream.single_output:
                return f"{var}.output"
            return f"{var}.outputs[{ref.output!r}]"

        rendered = {
            arg: render_input(
                value, params, pipeline.tasks, runtime_params=True, output_expr=output_expr
            )
            for arg, value in sorted(tspec.inputs.items())
        }
        var = f"{sanitize(tspec.name)}_task"
        call_args = ", ".join(f"{arg}={expr}" for arg, expr in rendered.items())
        emitter.line(f"{var} = {self._component_name(tspec, aliases)}({call_args})")
        emitter.line(f"{var}.set_display_name({tspec.name!r})")
        # KFP caches by default; emit explicitly to normalize to IR semantics.
        emitter.line(f"{var}.set_caching_options({tspec.cache!r})")
        if tspec.retry.retries > 0:
            emitter.line(
                f"{var}.set_retry(num_retries={tspec.retry.retries},"
                f" backoff_duration='{int(tspec.retry.delay_seconds)}s',"
                f" backoff_factor={tspec.retry.backoff!r})"
            )
        if tspec.resources.cpu:
            emitter.line(f"{var}.set_cpu_limit({tspec.resources.cpu!r})")
        if tspec.resources.memory:
            emitter.line(f"{var}.set_memory_limit({tspec.resources.memory!r})")
        if tspec.resources.gpu:
            emitter.line(f"{var}.set_accelerator_type({self._accelerator(tspec)!r})")
            emitter.line(f"{var}.set_accelerator_limit({tspec.resources.gpu})")

    def _accelerator(self, tspec: TaskSpec) -> str | None:
        return tspec.resources.gpu_type or self.options.get("default_accelerator")


_LITERAL_TYPES = {str: "str", bool: "bool", int: "int", float: "float", dict: "dict", list: "list"}


def _input_type(pipeline: Pipeline, tspec: TaskSpec, arg: str) -> str:
    """IR type flowing into an input, inferred from its source."""
    value = tspec.inputs[arg]
    if isinstance(value, ParamRef):
        return pipeline.params[value.param].type
    if isinstance(value, OutputRef):
        return pipeline.tasks[value.task].output_types.get(value.output, "Any")
    if isinstance(value, LiteralValue):
        # bool before int: bool is an int subclass.
        for pytype, name in _LITERAL_TYPES.items():
            if type(value.value) is pytype:
                return name
    return "Any"
