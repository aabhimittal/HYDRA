"""The ``hydraml`` command-line interface.

Subcommands:

- ``backends``  - list backends and their capability matrix
- ``inspect``   - print a pipeline's serialized IR
- ``graph``     - export the task graph (DOT or Mermaid)
- ``validate``  - report capability gaps for one or all backends
- ``compile``   - generate orchestrator-native code
- ``run``       - execute in-process with the local backend
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .backends import backend_names, get_backend
from .backends.base import _REGISTRY, Capability
from .backends.local import LocalBackend
from .errors import CompilationError, HydraError
from .ir.graph import to_dot, to_mermaid
from .ir.model import Pipeline
from .ir.serde import to_yaml
from .issues import errors, render_issues
from .loader import load_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except CompilationError as exc:
        if exc.issues:
            print(render_issues(exc.issues), file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except HydraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydraml",
        description="HYDRA: compile one ML pipeline definition to Kubeflow, Airflow, or Prefect.",
    )
    parser.add_argument("--version", action="version", version=f"hydraml {__version__}")
    parser.set_defaults(command=None)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("backends", help="list backends and capabilities")
    p.set_defaults(handler=_cmd_backends)

    p = sub.add_parser("inspect", help="print the pipeline IR as YAML")
    _add_spec(p)
    p.set_defaults(handler=_cmd_inspect)

    p = sub.add_parser("graph", help="export the task graph")
    _add_spec(p)
    p.add_argument("--format", choices=("dot", "mermaid"), default="dot")
    p.set_defaults(handler=_cmd_graph)

    p = sub.add_parser("validate", help="check a pipeline against backend capabilities")
    _add_spec(p)
    p.add_argument("--backend", default="all", help="backend name or 'all' (default)")
    p.set_defaults(handler=_cmd_validate)

    p = sub.add_parser("compile", help="generate orchestrator-native code")
    _add_spec(p)
    p.add_argument("--backend", required=True, choices=[n for n in backend_names() if n != "local"])
    p.add_argument("-o", "--outdir", default="build", help="output directory (default: build/)")
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a pipeline parameter (repeatable)",
    )
    p.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="backend option, e.g. --opt base_image=my/image (repeatable)",
    )
    p.set_defaults(handler=_cmd_compile)

    p = sub.add_parser("run", help="execute the pipeline in-process (local backend)")
    _add_spec(p)
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--no-cache", action="store_true", help="ignore and don't write the task cache")
    p.add_argument("--cache-dir", default=".hydra_cache")
    p.set_defaults(handler=_cmd_run)

    return parser


def _add_spec(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "spec",
        help="pipeline reference: module[:attr], path/to/file.py[:attr], or spec.yaml",
    )


# --- commands ---------------------------------------------------------------------


def _cmd_backends(args: argparse.Namespace) -> int:
    caps = list(Capability)
    width = max(len(c.value) for c in caps) + 2
    names = backend_names()
    print("capability".ljust(width) + "".join(n.ljust(10) for n in names))
    for cap in caps:
        row = cap.value.ljust(width)
        for name in names:
            row += ("yes" if cap in _REGISTRY[name].capabilities else "-").ljust(10)
        print(row)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    print(to_yaml(load_pipeline(args.spec)), end="")
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    pipe = load_pipeline(args.spec)
    print(to_dot(pipe) if args.format == "dot" else to_mermaid(pipe), end="")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    pipe = load_pipeline(args.spec)
    targets = backend_names() if args.backend == "all" else [args.backend]
    exit_code = 0
    for name in targets:
        issues = get_backend(name).validate(pipe)
        print(f"== {name} ==")
        if issues:
            print(render_issues(issues))
        else:
            print("clean: pipeline maps onto this backend with no loss")
        if errors(issues):
            exit_code = 1
        print()
    return exit_code


def _cmd_compile(args: argparse.Namespace) -> int:
    pipe = load_pipeline(args.spec)
    backend = get_backend(args.backend, **_parse_kv(args.opt))
    params = _coerce_params(pipe, _parse_kv(args.param))
    result = backend.compile(pipe, Path(args.outdir), params=params)
    warnings = [i for i in result.issues if i.severity.value != "error"]
    if warnings:
        print(render_issues(warnings), file=sys.stderr)
    for path in result.files:
        print(path)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    pipe = load_pipeline(args.spec)
    backend = LocalBackend(cache_dir=args.cache_dir)
    params = _coerce_params(pipe, _parse_kv(args.param))
    result = backend.run(pipe, params=params, use_cache=not args.no_cache)
    for name, run in result.task_runs.items():
        line = f"{run.status:9s} {name}"
        if run.status == "succeeded":
            line += f"  ({run.attempts} attempt{'s' if run.attempts != 1 else ''},"
            line += f" {run.duration_seconds:.2f}s)"
        elif run.status == "failed":
            line += f"  {run.error}"
        print(line)
    if result.succeeded:
        terminal = [
            name
            for name in result.task_runs
            if not any(name in t.upstream() for t in pipe.tasks.values())
        ]
        for name in terminal:
            try:
                rendered = json.dumps(result.outputs[name], indent=2, default=repr)
            except (TypeError, ValueError):
                rendered = repr(result.outputs[name])
            print(f"\noutput[{name}] = {rendered}")
        return 0
    return 1


# --- helpers -----------------------------------------------------------------------


def _parse_kv(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise HydraError(f"expected NAME=VALUE, got {pair!r}")
        out[key] = value
    return out


def _coerce_params(pipeline: Pipeline, raw: dict[str, str]) -> dict[str, Any]:
    """Coerce CLI string values using each param's declared IR type."""
    coerced: dict[str, Any] = {}
    for name, value in raw.items():
        spec = pipeline.params.get(name)
        if spec is None:
            raise HydraError(
                f"unknown parameter {name!r}"
                f" (pipeline has: {', '.join(pipeline.params) or 'none'})"
            )
        try:
            coerced[name] = _coerce(value, spec.type)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HydraError(f"parameter {name!r}: cannot parse {value!r} as {spec.type}") from exc
    return coerced


def _coerce(value: str, type_name: str) -> Any:
    if type_name == "int":
        return int(value)
    if type_name == "float":
        return float(value)
    if type_name == "bool":
        lowered = value.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(value)
    if type_name in ("dict", "list"):
        return json.loads(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
