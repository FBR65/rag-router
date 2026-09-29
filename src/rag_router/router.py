"""Router-Kernlogik (P4): Skip -> Renormalisieren -> Fan-out.

Nach dem Artikel-Design sind das zwei getrennte Entscheidungen:
1. p(none) >= skip -> gar nicht suchen (eine uebersprungene echte Frage
   bedeutet eine falsche Antwort; eine unnötige Suche ist billig).
2. Ansonsten Wahl NUR unter den RAGs (renormalisiert). Fuehrt ein RAG mit
   >= fanout -> allein durchsuchen; sonst Top-2 und der Answer Check
   entscheidet.

Grenzverhalten (dokumentierte Konstruktionsentscheidung):
- p(none) >= skip ist inklusiv: exakt 0.60 skippt, 0.5999 sucht.
- Beim Fan-out wird ohne Epsilon verglichen. Ein Kandidat im 1-ULP-Fenster
  unterhalb der Schwelle (z. B. 0.5499_999...9 aus 0.44/(0.44+0.36)) erzeugt
  Fan-out — das konservative Ergebnis: Fan-out sucht doppelt, aber überspringt
  nie falsch. Skip falsch herum wäre eine unbeantwortete echte Frage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from rag_router.decision.base import RouteDist
from rag_router.decision.laya import PredictFn  # noqa: F401  (Typ-Export)


@dataclass(frozen=True)
class DecisionThresholds:
    """Schwellwerte der Routing-Entscheidung (Artikel-Defaults)."""

    skip: float = 0.60
    fanout: float = 0.55
    answer: float = 0.50


@dataclass(frozen=True)
class RouteDecision:
    """Ergebnis der Routing-Entscheidung."""

    routes: list[str]
    distribution: Mapping[str, float]
    reason: str
    detail: Mapping[str, str] = field(default_factory=dict)


def _renormalize(
    probabilities: Mapping[str, float],
) -> list[tuple[str, float]]:
    """RAG-Kandidaten ohne 'none', renormalisiert, sortiert.

    Reihenfolge bei Gleichstand: Eingabe-Reihenfolge (stabil).
    """
    rag_probs = [(key, float(p)) for key, p in probabilities.items() if key != "none"]
    total = sum(p for _, p in rag_probs)
    if total <= 0.0:
        return []
    normalized = [(key, p / total) for key, p in rag_probs]
    # Sortierung nach p absteigend, Gleichstand = Eingabe-Reihenfolge stabil.
    order = sorted(
        range(len(normalized)),
        key=lambda i: (-normalized[i][1], i),
    )
    return [normalized[i] for i in order]


class RouterDecisionEngine:
    """Fuehrt die zweistufige Routing-Entscheidung aus."""

    def __init__(self, backend, thresholds: DecisionThresholds | None = None):
        self._backend = backend
        self._thresholds = thresholds or DecisionThresholds()

    @property
    def thresholds(self) -> DecisionThresholds:
        return self._thresholds

    def route(self, question: str) -> RouteDecision:
        dist: RouteDist = self._backend.decide(question, self._rag_descriptions)
        probabilities = dict(dist.probabilities)

        if not probabilities:
            raise ValueError("Wahrscheinlichkeitsverteilung ist leer")
        negatives = [k for k, v in probabilities.items() if v < 0.0]
        if negatives:
            raise ValueError(f"negative Wahrscheinlichkeiten für {sorted(negatives)}")

        p_none = float(probabilities.get("none", 0.0))
        if p_none >= self._thresholds.skip:
            return RouteDecision(
                routes=["none"],
                distribution=probabilities,
                reason="skip",
                detail={"p_none": f"{p_none:.4f}"},
            )

        ranked = _renormalize(probabilities)
        if not ranked:
            # Keine RAG-Kandidaten mit Masse: fail closed auf Suche unmöglich,
            # also none (kann nur bei p_rag_total == 0 bei skip < p_none).
            return RouteDecision(
                routes=["none"],
                distribution=probabilities,
                reason="no_rag_mass",
            )
        if ranked[0][1] < self._thresholds.fanout and len(ranked) >= 2:
            routes = [ranked[0][0], ranked[1][0]]
            reason = "fanout"
        else:
            routes = [ranked[0][0]]
            reason = "clear_leader"
        return RouteDecision(
            routes=routes,
            distribution=probabilities,
            reason=reason,
            detail={
                "p_none": f"{p_none:.4f}",
                "kb_share": f"{ranked[0][1]:.4f}",
            },
        )

    # -- Konfiguration ---------------------------------------------------

    def set_rag_descriptions(
        self, descriptions: Mapping[str, str]
    ) -> RouterDecisionEngine:
        self._rag_descriptions = descriptions
        return self

    _rag_descriptions: Mapping[str, str] = {}
