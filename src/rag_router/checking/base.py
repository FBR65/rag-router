"""Answer-Check Protocols und DTOs (P6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class CheckResult:
    """Ergebnis des Answer-Checks."""

    p_answered: float
    raw: Any = field(default=None, repr=False, compare=False)
    error: str | None = None
    source: str = "probabilities"


@runtime_checkable
class AnswerChecker(Protocol):
    """Prueft: beantworten die Passagen die Frage?"""

    def check(self, question: str, passages: Sequence[str]) -> CheckResult: ...
