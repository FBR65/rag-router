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


@dataclass(frozen=True)
class SkipDist:
    """Ergebnis der Skip-Entscheidung (D1).

    p_recall = P(Dokumentrecherche noetig); p_none = 1 - p_recall.
    """

    p_recall: float
    raw: Any = field(default=None, repr=False, compare=False)
    source: str = "probabilities"
    detail: Mapping[str, str] = field(default_factory=dict)

    @property
    def p_none(self) -> float:
        return 1.0 - self.p_recall


@dataclass(frozen=True)
class KbDist:
    """Ergebnis der KB-Wahl (D2): Verteilung NUR ueber die RAG-Keys.

    Enthaelt bewusst keinen 'none'-Key — die Skip-Entscheidung ist davon
    getrennt (D1). Summe der Werte ~1.
    """

    probabilities: Mapping[str, float]
    raw: Any = field(default=None, repr=False, compare=False)
    source: str = "probabilities"
    detail: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class DecisionRoute(Protocol):
    """Ein Weg zur Entscheidung: D1 (skip) und D2 (choose) nativ getrennt."""

    name: str

    def skip(
        self, question: str, rag_descriptions: Mapping[str, str] | None = None
    ) -> SkipDist: ...

    def choose(
        self, question: str, rag_descriptions: Mapping[str, str]
    ) -> KbDist: ...




@runtime_checkable
class DecisionBackend(Protocol):
    """Ein Backend, das Fragen auf Routen verteilt."""

    def decide(
        self, question: str, rag_descriptions: Mapping[str, str]
    ) -> RouteDist: ...
