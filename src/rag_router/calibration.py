"""Kalibrierung der Auto-Werte (P11).

Reine Logik: aus gelabelten Messpunkten werden Schwellen, Cascade-Grauzone und
der beste Entscheidungsweg abgeleitet (docs/spec-decision-stages.md, §2.3).
Kein Modellaufruf, keine Persistenz hier — das macht die Pipeline.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

GENERATOR_VERSION = 1

# Dokumentierte Fallbacks, wenn nicht genug/keine Messpunkte vorliegen.
FALLBACK_SKIP = 0.60
FALLBACK_FANOUT = 0.55
FALLBACK_ANSWER = 0.50
FALLBACK_CASCADE = (0.40, 0.60)


@dataclass(frozen=True)
class CalibrationSample:
    """Ein Messpunkt einer Route fuer eine Kalibrierungsfrage."""

    question_id: str
    gold_route: str                 # "none" oder ein RAG-Key
    answerable: bool
    skip_p_none: float              # p(none) aus D1 der Route
    kb: Mapping[str, float]         # renormalisierte KB-Verteilung (D2)
    p_answered: float | None = None # Answer-Check, falls gelaufen


@dataclass(frozen=True)
class DerivedThresholds:
    skip: float
    fanout: float
    answer: float
    cascade_lo: float
    cascade_hi: float
    sources: Mapping[str, int] = field(default_factory=dict)


def _min_no_retrieval(samples: Sequence[CalibrationSample]) -> float | None:
    values = [s.skip_p_none for s in samples if s.gold_route == "none"]
    return min(values) if values else None


def _max_answerable_nonretrieval(samples: Sequence[CalibrationSample]) -> float | None:
    values = [
        s.skip_p_none
        for s in samples
        if s.gold_route != "none" and s.answerable
    ]
    return max(values) if values else None


def _skip_threshold(samples: Sequence[CalibrationSample]) -> float:
    min_no = _min_no_retrieval(samples)
    if min_no is None:
        return FALLBACK_SKIP
    max_ans = _max_answerable_nonretrieval(samples)
    if max_ans is None:
        # Keine beantwortbaren Fragen: konservativ (moeglichst wenig skippen).
        return 0.0
    if max_ans < min_no:
        # Separierbar. Schwelle KNAPP ueber der hoechsten echten Frage, nicht in
        # der Lueckenmitte: die Mitte ist empfindlich gegen einen einzelnen
        # Ausreisser knapp unter min_no und skippt dann echte Fragen mit.
        # Prioritaet (Spec §1): eine echte Frage darf nie geskippt werden,
        # eine unnoetige Suche ist billig.
        return max_ans + 0.10 * (min_no - max_ans)
    # Nicht separierbar: min_no (Prioritaet: no-retrieval skippt).
    return min_no


def _fanout_threshold(samples: Sequence[CalibrationSample]) -> float:
    leaders: list[float] = []
    for s in samples:
        if s.gold_route == "none" or not s.answerable:
            continue
        if s.gold_route not in s.kb:
            continue
        leader = max(s.kb.values(), default=0.0)
        # Nur wenn die Gold-KB die Fuehrung hat, ist Fanout steuerbar.
        if s.kb.get(s.gold_route, 0.0) >= leader:
            leaders.append(leader)
    if not leaders:
        return FALLBACK_FANOUT
    # Fuehrung >= fanout -> allein durchsuchen. fanout minimal halten, damit
    # auch knappe Fuehrungen nicht faelschlich fan-outen.
    return max(0.0, min(leaders) - 0.001)


def _answer_threshold(samples: Sequence[CalibrationSample]) -> float:
    answered = [
        s.p_answered
        for s in samples
        if s.answerable and s.p_answered is not None
    ]
    not_answered = [
        s.p_answered
        for s in samples
        if not s.answerable and s.gold_route != "none" and s.p_answered is not None
    ]
    if not answered:
        return FALLBACK_ANSWER
    if not not_answered:
        return min(answered)
    lo = max(not_answered)
    hi = min(answered)
    if lo < hi:
        return (lo + hi) / 2.0
    return hi  # nicht separierbar: Prioritaet beantwortbar


def _cascade(samples: Sequence[CalibrationSample]) -> tuple[float, float]:
    leaders = [max(s.kb.values(), default=0.0) for s in samples if s.kb]
    if not leaders:
        return FALLBACK_CASCADE
    median = statistics.median(leaders)
    lo = max(0.0, median - 0.10)
    hi = min(1.0, median + 0.10)
    return (lo, hi)


def derive(samples: Sequence[CalibrationSample]) -> DerivedThresholds:
    """Leitet alle Auto-Schwellen aus Messpunkten ab (dokumentierte Fallbacks)."""
    lo, hi = _cascade(samples)
    sources = {
        "questions": len(samples),
        "no_retrieval": sum(1 for s in samples if s.gold_route == "none"),
        "answerable": sum(1 for s in samples if s.answerable),
    }
    return DerivedThresholds(
        skip=_skip_threshold(samples),
        fanout=_fanout_threshold(samples),
        answer=_answer_threshold(samples),
        cascade_lo=lo,
        cascade_hi=hi,
        sources=sources,
    )


def route_accuracy(
    samples: Sequence[CalibrationSample], thresholds: DerivedThresholds
) -> float:
    """Trefferquote einer Route: Skip und (falls Gold bekannt) KB-Wahl.

    ``gold_route == "none"``  -> korrekt, wenn geskippt.
    ``gold_route == ""``      -> Recherche erwartet, KB unbekannt: korrekt, wenn
    nicht geskippt (generischer Kalibrierungssatz ohne KB-Gold).
    sonst                     -> KB-Wahl gegen Gold.
    """
    if not samples:
        return 0.0
    hits = 0
    for s in samples:
        if s.gold_route == "none":
            if s.skip_p_none >= thresholds.skip:
                hits += 1
            continue
        if s.skip_p_none >= thresholds.skip:
            continue  # faelschlich geskippt
        if s.gold_route == "":
            hits += 1  # nur die Skip-Entscheidung bewertbar
            continue
        leader = max(s.kb, key=lambda k: s.kb[k], default=None)
        if leader is None:
            continue
        if s.kb.get(leader, 0.0) >= thresholds.fanout:
            chosen = [leader]
        else:
            ranked = sorted(s.kb, key=lambda k: s.kb[k], reverse=True)
            chosen = ranked[:2]
        if s.gold_route in chosen:
            hits += 1
    return hits / len(samples)


@dataclass(frozen=True)
class RouteScore:
    name: str
    accuracy: float


def choose_best_route(
    route_samples: Mapping[str, Sequence[CalibrationSample]],
    route_thresholds: Mapping[str, DerivedThresholds],
) -> RouteScore:
    """Bester Weg = hoechste Trefferquote; Gleichstand: Reihenfolge (stabil)."""
    best: RouteScore | None = None
    for name, samples in route_samples.items():
        thr = route_thresholds[name]
        score = RouteScore(name=name, accuracy=route_accuracy(samples, thr))
        if best is None or score.accuracy > best.accuracy:
            best = score
    if best is None:
        return RouteScore(name="laya", accuracy=0.0)
    return best


@dataclass(frozen=True)
class AnswerBackendScore:
    name: str
    accuracy: float


def choose_best_answer_backend(
    scores: Mapping[str, float],
) -> AnswerBackendScore:
    if not scores:
        return AnswerBackendScore(name="laya", accuracy=0.0)
    name = max(scores, key=lambda k: scores[k])
    return AnswerBackendScore(name=name, accuracy=scores[name])


@dataclass
class CalibrationProfile:
    """Ergebnis der Messung; versioniert und reproduzierbar."""

    decision_route: str
    answer_backend: str
    skip: float
    fanout: float
    answer: float
    cascade_lo: float
    cascade_hi: float
    generator_version: int = GENERATOR_VERSION
    created_at: str = ""
    sources: Mapping[str, Any] = field(default_factory=dict)
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_version": self.generator_version,
            "created_at": self.created_at,
            "decision_route": self.decision_route,
            "answer_backend": self.answer_backend,
            "thresholds": {
                "skip": self.skip,
                "fanout": self.fanout,
                "answer": self.answer,
            },
            "cascade_lo": self.cascade_lo,
            "cascade_hi": self.cascade_hi,
            "sources": dict(self.sources),
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationProfile:
        thr = data.get("thresholds") or {}
        return cls(
            decision_route=str(data.get("decision_route", "laya")),
            answer_backend=str(data.get("answer_backend", "laya")),
            skip=float(thr.get("skip", FALLBACK_SKIP)),
            fanout=float(thr.get("fanout", FALLBACK_FANOUT)),
            answer=float(thr.get("answer", FALLBACK_ANSWER)),
            cascade_lo=float(data.get("cascade_lo", FALLBACK_CASCADE[0])),
            cascade_hi=float(data.get("cascade_hi", FALLBACK_CASCADE[1])),
            generator_version=int(data.get("generator_version", 0)),
            created_at=str(data.get("created_at", "")),
            sources=data.get("sources") or {},
            fallback=bool(data.get("fallback", False)),
        )


def fallback_profile(decision_route: str = "laya", answer_backend: str = "laya") -> CalibrationProfile:
    """Dokumentierter Kaltstart, wenn keine Messung moeglich ist."""
    return CalibrationProfile(
        decision_route=decision_route,
        answer_backend=answer_backend,
        skip=FALLBACK_SKIP,
        fanout=FALLBACK_FANOUT,
        answer=FALLBACK_ANSWER,
        cascade_lo=FALLBACK_CASCADE[0],
        cascade_hi=FALLBACK_CASCADE[1],
        sources={"mode": "fallback"},
        fallback=True,
    )
