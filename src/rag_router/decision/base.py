"""Decision-Protocols und DTOs (P3)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

NO_RETRIEVAL_KEY = "none"

# Feste Beschreibung der No-Retrieval-Route (Artikel: "Not a question about
# our documents: small talk, maths, jokes, writing or coding help").
NO_RETRIEVAL_DESCRIPTION = (
    "Keine Dokument-Recherche nötig: Small Talk, Mathe, Witze, "
    "Schreib- oder Programmierhilfe, allgemeines Weltwissen"
)


@dataclass(frozen=True)
class RouteDist:
    """Wahrscheinlichkeiten je Route (Key -> p in [0, 1])."""

    probabilities: Mapping[str, float]
    raw: Any = field(default=None, repr=False, compare=False)
    probabilities_source: str = "probabilities"


@runtime_checkable
class DecisionBackend(Protocol):
    """Ein Backend, das Fragen auf Routen verteilt."""

    def decide(
        self, question: str, rag_descriptions: Mapping[str, str]
    ) -> RouteDist: ...
