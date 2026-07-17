"""Resolve pipeline spec strings to Pipeline objects.

Spec forms:

- ``pkg.module:attr``      - import a module, take the named Pipeline
- ``pkg.module``           - import a module, find the single Pipeline in it
- ``path/to/file.py:attr`` - load a file as a module
- ``path/to/file.py``      - load a file, find the single Pipeline
- ``path/to/spec.yaml``    - deserialize a serialized IR document

For ``.py`` file paths the module is registered in ``sys.modules`` under its
dotted name relative to the CWD when one can be derived (so fn_refs recorded
at decoration time are importable, and codegen emits real import paths).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .errors import LoadError
from .ir.model import Pipeline
from .ir.serde import from_yaml


def load_pipeline(spec: str) -> Pipeline:
    target, _, attr = spec.partition(":")
    path = Path(target)
    if target.endswith((".yaml", ".yml")):
        if attr:
            raise LoadError(f"YAML specs do not take an :attr suffix: {spec!r}")
        try:
            return from_yaml(path.read_text())
        except OSError as exc:
            raise LoadError(f"cannot read {target!r}: {exc}") from exc
    if target.endswith(".py"):
        module = _load_file(path)
    else:
        # Console scripts don't get the CWD on sys.path the way `python -m`
        # does; add it so `hydraml run my_project.pipelines:p` works from a
        # project root.
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise LoadError(f"cannot import module {target!r}: {exc}") from exc
    return _pick_pipeline(module, attr, spec)


def module_name_for(path: Path) -> str | None:
    """Dotted module name for a file relative to CWD, if derivable."""
    try:
        rel = path.resolve().relative_to(Path.cwd())
    except ValueError:
        return None
    parts = [*rel.parts[:-1], rel.stem]
    if all(part.isidentifier() for part in parts):
        return ".".join(parts)
    return None


def _load_file(path: Path) -> ModuleType:
    if not path.exists():
        raise LoadError(f"no such file: {path}")
    name = module_name_for(path) or f"_hydra_loaded_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LoadError(f"cannot load {path} as a module")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so decorators see the final module name and the
    # local runner / generated code can re-import task functions.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _pick_pipeline(module: ModuleType, attr: str, spec: str) -> Pipeline:
    if attr:
        obj = getattr(module, attr, None)
        if obj is None:
            raise LoadError(f"module {module.__name__!r} has no attribute {attr!r}")
        if not isinstance(obj, Pipeline):
            raise LoadError(
                f"{spec!r} is a {type(obj).__name__}, not a Pipeline - is it"
                " decorated with @pipeline?"
            )
        return obj
    pipelines = {
        name: obj for name, obj in vars(module).items() if isinstance(obj, Pipeline)
    }
    if not pipelines:
        raise LoadError(f"no @pipeline definitions found in {module.__name__!r}")
    if len(pipelines) > 1:
        raise LoadError(
            f"multiple pipelines in {module.__name__!r}"
            f" ({', '.join(sorted(pipelines))}); disambiguate with"
            f" {spec}:<name>"
        )
    return next(iter(pipelines.values()))
