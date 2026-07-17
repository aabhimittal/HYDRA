"""HYDRA - one pipeline definition, many orchestrators.

HYDRA is a poly-orchestration compiler for ML pipelines: author a pipeline
once against a small backend-neutral IR, then validate and compile it to
Kubeflow Pipelines, Airflow, or Prefect - or execute it in-process with the
local backend.

(The distribution/import name is ``hydraml`` to stay clear of ``hydra-core``,
the configuration framework, which owns the ``hydra`` import namespace.)
"""

__version__ = "0.1.0"

from .backends import Capability, backend_names, get_backend  # noqa: E402
from .backends.local import LocalBackend, RunResult  # noqa: E402
from .dsl import pipeline, task  # noqa: E402
from .errors import HydraError, UsageError  # noqa: E402
from .ir import Pipeline, Resources, RetryPolicy  # noqa: E402
from .loader import load_pipeline  # noqa: E402

__all__ = [
    "Capability",
    "HydraError",
    "LocalBackend",
    "Pipeline",
    "Resources",
    "RetryPolicy",
    "RunResult",
    "UsageError",
    "__version__",
    "backend_names",
    "get_backend",
    "load_pipeline",
    "pipeline",
    "task",
]
