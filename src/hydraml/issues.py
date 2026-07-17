"""Validation issues.

Issues are how HYDRA reports the gap between what a pipeline asks for and what
a given backend can actually deliver. ERROR blocks compilation; WARNING means
the feature degrades or is dropped; INFO records a semantic difference the
author should know about (e.g. params bound at compile time).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    task: str | None = None

    def render(self) -> str:
        location = f" [task: {self.task}]" if self.task else ""
        return f"{self.severity.value.upper():7s} {self.code}: {self.message}{location}"


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity is Severity.ERROR]


def render_issues(issues: list[Issue]) -> str:
    return "\n".join(issue.render() for issue in issues)
