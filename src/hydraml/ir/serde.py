"""Serialization of the IR to/from plain dicts and YAML.

The serialized form is the stable interface for tooling: diffing pipeline
versions in review, generating docs, or feeding future non-Python frontends.
"""

from __future__ import annotations

from typing import Any

import yaml

from .model import (
    REQUIRED,
    SCHEMA_VERSION,
    InputValue,
    LiteralValue,
    OutputRef,
    ParamRef,
    ParamSpec,
    Pipeline,
    Resources,
    RetryPolicy,
    TaskSpec,
)


def _encode_input(value: InputValue) -> dict[str, Any]:
    if isinstance(value, ParamRef):
        return {"kind": "param", "param": value.param}
    if isinstance(value, OutputRef):
        return {"kind": "output", "task": value.task, "output": value.output}
    return {"kind": "literal", "value": value.value}


def _decode_input(data: dict[str, Any]) -> InputValue:
    kind = data.get("kind")
    if kind == "param":
        return ParamRef(param=data["param"])
    if kind == "output":
        return OutputRef(task=data["task"], output=data.get("output", "result"))
    if kind == "literal":
        return LiteralValue(value=data["value"])
    raise ValueError(f"unknown input kind: {kind!r}")


def to_dict(pipeline: Pipeline) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": pipeline.name,
    }
    if pipeline.description:
        doc["description"] = pipeline.description
    if pipeline.schedule:
        doc["schedule"] = pipeline.schedule
    if pipeline.tags:
        doc["tags"] = list(pipeline.tags)
    doc["params"] = {
        name: _encode_param(spec) for name, spec in pipeline.params.items()
    }
    doc["tasks"] = {name: _encode_task(task) for name, task in pipeline.tasks.items()}
    return doc


def _encode_param(spec: ParamSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"type": spec.type}
    if not spec.required:
        out["default"] = spec.default
    if spec.description:
        out["description"] = spec.description
    return out


def _encode_task(task: TaskSpec) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fn": task.fn_ref,
        "inputs": {k: _encode_input(v) for k, v in task.inputs.items()},
    }
    if task.outputs != ("result",):
        out["outputs"] = list(task.outputs)
    if task.output_types:
        out["output_types"] = dict(task.output_types)
    if task.retry != RetryPolicy():
        out["retry"] = {
            "max_attempts": task.retry.max_attempts,
            "delay_seconds": task.retry.delay_seconds,
            "backoff": task.retry.backoff,
        }
    if not task.resources.is_empty:
        res: dict[str, Any] = {}
        if task.resources.cpu:
            res["cpu"] = task.resources.cpu
        if task.resources.memory:
            res["memory"] = task.resources.memory
        if task.resources.gpu:
            res["gpu"] = task.resources.gpu
            if task.resources.gpu_type:
                res["gpu_type"] = task.resources.gpu_type
        out["resources"] = res
    if task.cache:
        out["cache"] = True
    if task.description:
        out["description"] = task.description
    return out


def from_dict(doc: dict[str, Any]) -> Pipeline:
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r} (expected {SCHEMA_VERSION})")
    params = {
        name: ParamSpec(
            name=name,
            type=spec.get("type", "Any"),
            default=spec.get("default", REQUIRED) if "default" in spec else REQUIRED,
            description=spec.get("description"),
        )
        for name, spec in (doc.get("params") or {}).items()
    }
    tasks = {}
    for name, tdoc in (doc.get("tasks") or {}).items():
        retry = tdoc.get("retry") or {}
        res = tdoc.get("resources") or {}
        tasks[name] = TaskSpec(
            name=name,
            fn_ref=tdoc["fn"],
            inputs={k: _decode_input(v) for k, v in (tdoc.get("inputs") or {}).items()},
            outputs=tuple(tdoc.get("outputs", ("result",))),
            output_types=dict(tdoc.get("output_types") or {}),
            retry=RetryPolicy(
                max_attempts=retry.get("max_attempts", 1),
                delay_seconds=retry.get("delay_seconds", 0.0),
                backoff=retry.get("backoff", 1.0),
            ),
            resources=Resources(
                cpu=res.get("cpu"),
                memory=res.get("memory"),
                gpu=res.get("gpu", 0),
                gpu_type=res.get("gpu_type"),
            ),
            cache=bool(tdoc.get("cache", False)),
            description=tdoc.get("description"),
        )
    return Pipeline(
        name=doc["name"],
        description=doc.get("description"),
        schedule=doc.get("schedule"),
        params=params,
        tasks=tasks,
        tags=tuple(doc.get("tags") or ()),
    )


def to_yaml(pipeline: Pipeline) -> str:
    return yaml.safe_dump(to_dict(pipeline), sort_keys=False, default_flow_style=False)


def from_yaml(text: str) -> Pipeline:
    return from_dict(yaml.safe_load(text))
