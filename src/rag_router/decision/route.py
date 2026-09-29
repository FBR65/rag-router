"""Entscheidungswege (P10): D1 (skip) und D2 (choose) je Weg nativ getrennt.

Drei Wege: LayaRoute (Laya-Checkpoint), SlmRoute (OpenAI-kompatibles SLM),
HybridRoute (primary + secondary, cascade|committee). LegacyRoute adaptiert
das alte DecisionBackend (eine Verteilung inkl. 'none') auf den neuen Vertrag.

D1 und D2 sind je Weg getrennt: die Skip-Entscheidung kennt die KBs nicht, die
KB-Wahl kennt 'none' nicht. Dadurch kann eine grosse 'none'-Masse keine
KB-Masse mehr verdraengen (siehe docs/spec-decision-stages.md, §1).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from rag_router.decision.base import (
    NO_RETRIEVAL_DESCRIPTION,
    KbDist,
    RouteDist,
    SkipDist,
)

PredictFn = Callable[..., Mapping[str, Any]]

# --- Laya-Fragen ----------------------------------------------------------- #

SKIP_QUESTION_KEY = "needs_docs"
SKIP_INSTRUCTIONS = (
    "Bezieht sich die Frage auf Dokumente und interne Regeln der "
    "konfigurierten Wissensbasen?"
)
KB_QUESTION_KEY = "route"
KB_INSTRUCTIONS = "Welche Wissensbasis kann diese Frage beantworten?"


def build_skip_question() -> dict[str, Any]:
    """Noul-Frage (D1): p(true) = P(Dokumentrecherche noetig)."""
    return {
        SKIP_QUESTION_KEY: {
            "type": "noul",
            "instructions": SKIP_INSTRUCTIONS,
            "criteria": {
                "true": "ja, eine Dokumentenfrage",
                "false": "nein, Small Talk, Mathe, Allgemeinwissen, Schreib-/Programmierhilfe",
            },
        }
    }


def build_kb_question(rag_descriptions: Mapping[str, str]) -> dict[str, Any]:
    """Choice-Frage (D2) NUR ueber die RAG-Keys (kein 'none')."""
    return {
        KB_QUESTION_KEY: {
            "type": "choice",
            "instructions": KB_INSTRUCTIONS,
            "criteria": {key: desc for key, desc in rag_descriptions.items()},
        }
    }


def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    best = max(scores.values())
    total = sum(math.exp(s - best) for s in scores.values())
    return {k: math.exp(v - best) / total for k, v in scores.items()}


def _renormalize(probabilities: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in probabilities.values())
    if total <= 0.0:
        return {k: 0.0 for k in probabilities}
    return {k: float(v) / total for k, v in probabilities.items()}


class LayaRoute:
    """D1 = noul-Frage, D2 = choice ohne 'none' (je ein predict-Call)."""

    name = "laya"

    def __init__(
        self,
        predict: PredictFn,
        *,
        model: str = "multilingual",
        max_len: int = 1024,
    ) -> None:
        self._predict = predict
        self._model = model
        self._max_len = max_len

    @classmethod
    def from_laya(
        cls,
        *,
        model: str = "multilingual",
        max_len: int = 1024,
        preload: bool = False,
    ) -> LayaRoute:
        try:
            from laya import Router
        except ImportError as error:  # pragma: no cover - Umgebungsabhaengig
            raise RuntimeError(
                "laya route benoetigt das Paket 'laya': uv add laya"
            ) from error
        router = Router(preload=preload, auto_task_detection=False)
        return cls(predict=router.predict, model=model, max_len=max_len)

    @classmethod
    def from_predict(
        cls,
        predict: PredictFn,
        *,
        model: str = "multilingual",
        max_len: int = 1024,
    ) -> LayaRoute:
        return cls(predict=predict, model=model, max_len=max_len)

    def skip(
        self, question: str, rag_descriptions: Mapping[str, str] | None = None
    ) -> SkipDist:
        payload = self._predict(
            question,
            build_skip_question(),
            model=self._model,
            max_len=self._max_len,
        )
        answer = (payload.get("answers") or {}).get(SKIP_QUESTION_KEY) or {}
        raw = answer.get("noul")
        if not isinstance(raw, (int, float)):
            raise TypeError(
                "Laya lieferte keine noul-Wahrscheinlichkeit fuer die Skip-Frage"
            )
        p_recall = 1.0 - float(raw)
        return SkipDist(p_recall=p_recall, raw=payload, source="noul")

    def choose(
        self, question: str, rag_descriptions: Mapping[str, str]
    ) -> KbDist:
        payload = self._predict(
            question,
            build_kb_question(rag_descriptions),
            model=self._model,
            max_len=self._max_len,
        )
        answer = (payload.get("answers") or {}).get(KB_QUESTION_KEY) or {}
        raw_probs = answer.get("probabilities") or {}
        probabilities = {
            key: max(0.0, float(raw_probs.get(key, 0.0)))
            for key in rag_descriptions
        }
        return KbDist(
            probabilities=_renormalize(probabilities),
            raw=payload,
            source="probabilities",
        )


# --- SLM -------------------------------------------------------------------- #

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SKIP_LABELS = {"recall": "A", "none": "B"}


def _build_slm_skip_prompt(question: str) -> str:
    return (
        "Bezieht sich die folgende Frage auf Dokumente und interne Regeln der "
        "konfigurierten Wissensbasen, oder ist sie Small Talk, Mathematik, "
        "Allgemeinwissen oder Schreib-/Programmierhilfe?\n\n"
        "A = Ja, eine Dokumentenfrage; es lohnt sich, die Wissensbasen zu durchsuchen.\n"
        "B = Nein, keine Dokumentenfrage; Recherche ist unnoetig.\n\n"
        f"Frage: {question}\n\n"
        "Antworte ausschliesslich mit dem Buchstaben."
    )


def _build_slm_kb_prompt(question: str, rag_descriptions: Mapping[str, str]) -> str:
    lines = [
        f"{_LABELS[i]}. {key}: {description}"
        for i, (key, description) in enumerate(rag_descriptions.items())
    ]
    return (
        f"{KB_INSTRUCTIONS} Waehle genau eine Option.\n\n"
        f"Frage: {question}\n\n" + "\n".join(lines) + "\n\n"
        "Antworte ausschliesslich mit dem Buchstaben der Option."
    )


class SlmRoute:
    """D1 = yes/no-logprobs-Prompt, D2 = A/B/...-Prompt (je ein Call)."""

    name = "slm"

    def __init__(self, client: Any, *, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_settings(
        cls, *, base_url: str, api_key: str, model: str
    ) -> SlmRoute:
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - Umgebungsabhaengig
            raise RuntimeError("slm route benoetigt das Paket 'openai'") from error
        return cls(OpenAI(base_url=base_url, api_key=api_key, max_retries=0), model=model)

    @classmethod
    def from_client(cls, client: Any, *, model: str) -> SlmRoute:
        return cls(client, model=model)

    def _logprobs(self, prompt: str, labels: Mapping[str, str]) -> dict[str, float] | None:
        valid = set(labels.values())
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
        choice = response.choices[0]
        logprobs = choice.logprobs
        if logprobs is None or not logprobs.content:
            return None
        scores: dict[str, float] = {}
        for entry in logprobs.content[0].top_logprobs or ():
            token = entry.token.strip()
            if token in valid:
                scores[token] = max(scores.get(token, entry.logprob), entry.logprob)
        if not scores:
            return None
        return _softmax(scores)

    def _text_label(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
        )
        content = (response.choices[0].message.content or "").strip()
        match = re.search(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", content)
        return match.group(1) if match else ""

    def skip(
        self, question: str, rag_descriptions: Mapping[str, str] | None = None
    ) -> SkipDist:
        prompt = _build_slm_skip_prompt(question)
        probs = self._logprobs(prompt, _SKIP_LABELS)
        if probs is not None:
            p_recall = float(probs.get(_SKIP_LABELS["recall"], 0.0))
            return SkipDist(p_recall=p_recall, raw=probs, source="logprobs")
        label = self._text_label(prompt)
        p_recall = 1.0 if label == _SKIP_LABELS["recall"] else 0.0
        return SkipDist(p_recall=p_recall, raw=label, source="text")

    def choose(
        self, question: str, rag_descriptions: Mapping[str, str]
    ) -> KbDist:
        labels = {
            key: _LABELS[i] for i, key in enumerate(rag_descriptions)
        }
        label_to_key = {v: k for k, v in labels.items()}
        prompt = _build_slm_kb_prompt(question, rag_descriptions)
        probs = self._logprobs(prompt, labels)
        if probs is not None:
            probabilities = {
                key: float(probs.get(label, 0.0)) for key, label in labels.items()
            }
            return KbDist(
                probabilities=_renormalize(probabilities),
                raw=probs,
                source="logprobs",
            )
        label = self._text_label(prompt)
        key = label_to_key.get(label)
        probabilities = {k: 0.0 for k in rag_descriptions}
        if key is None:
            key = next(iter(rag_descriptions))  # fail closed auf die erste KB
        probabilities[key] = 1.0
        return KbDist(probabilities=probabilities, raw=label, source="text")


# --- Hybrid ----------------------------------------------------------------- #

AGGREGATES = ("mean", "product", "max", "primary")


class HybridRoute:
    """primary entscheidet; secondary greift je Strategie.

    cascade: secondary nur in der Grauzone [lo, hi].
    committee: beide, aggregiert per mean|product|max|primary.
    """

    name = "hybrid"

    def __init__(
        self,
        primary,
        secondary,
        *,
        strategy: str = "cascade",
        aggregate: str = "mean",
        cascade_lo: float = 0.40,
        cascade_hi: float = 0.60,
    ) -> None:
        if strategy not in ("cascade", "committee"):
            raise ValueError(f"unbekannte hybrid strategy: {strategy!r}")
        if aggregate not in AGGREGATES:
            raise ValueError(f"unbekannte hybrid aggregate: {aggregate!r}")
        if not 0.0 <= cascade_lo <= cascade_hi <= 1.0:
            raise ValueError(
                f"cascade-Grauzone ungueltig: [{cascade_lo}, {cascade_hi}]"
            )
        self._primary = primary
        self._secondary = secondary
        self._strategy = strategy
        self._aggregate = aggregate
        self._cascade_lo = cascade_lo
        self._cascade_hi = cascade_hi

    @property
    def strategy(self) -> str:
        return self._strategy

    def _in_gray(self, value: float) -> bool:
        return self._cascade_lo <= value <= self._cascade_hi

    def _combine(self, base: Mapping[str, float], other: Mapping[str, float]) -> dict[str, float]:
        keys = list(base)
        if self._aggregate == "primary":
            return {k: float(base.get(k, 0.0)) for k in keys}
        if self._aggregate == "mean":
            return {
                k: (float(base.get(k, 0.0)) + float(other.get(k, 0.0))) / 2.0
                for k in keys
            }
        if self._aggregate == "product":
            return {
                k: float(base.get(k, 0.0)) * float(other.get(k, 0.0)) for k in keys
            }
        # max
        return {
            k: max(float(base.get(k, 0.0)), float(other.get(k, 0.0))) for k in keys
        }

    def skip(
        self, question: str, rag_descriptions: Mapping[str, str] | None = None
    ) -> SkipDist:
        first = self._primary.skip(question, rag_descriptions)
        if self._strategy == "committee":
            second = self._secondary.skip(question, rag_descriptions)
            combined = self._combine(
                {"recall": first.p_recall}, {"recall": second.p_recall}
            )
            return SkipDist(
                p_recall=combined["recall"],
                raw={"primary": first.raw, "secondary": second.raw},
                source=f"committee:{self._aggregate}",
                detail={
                    self._primary.name: f"{first.p_recall:.4f}",
                    self._secondary.name: f"{second.p_recall:.4f}",
                },
            )
        if self._in_gray(first.p_recall):
            second = self._secondary.skip(question, rag_descriptions)
            return SkipDist(
                p_recall=second.p_recall,
                raw={"primary": first.raw, "secondary": second.raw},
                source=f"cascade:{self._secondary.name}",
                detail={
                    self._primary.name: f"{first.p_recall:.4f}",
                    self._secondary.name: f"{second.p_recall:.4f}",
                },
            )
        return SkipDist(
            p_recall=first.p_recall,
            raw=first.raw,
            source=f"cascade:{self._primary.name}",
            detail={self._primary.name: f"{first.p_recall:.4f}"},
        )

    def choose(
        self, question: str, rag_descriptions: Mapping[str, str]
    ) -> KbDist:
        first = self._primary.choose(question, rag_descriptions)
        if self._strategy == "committee":
            second = self._secondary.choose(question, rag_descriptions)
            combined = self._combine(
                dict(first.probabilities), dict(second.probabilities)
            )
            return KbDist(
                probabilities=_renormalize(combined),
                raw={"primary": first.raw, "secondary": second.raw},
                source=f"committee:{self._aggregate}",
                detail={
                    self._primary.name: ",".join(
                        f"{k}={v:.3f}" for k, v in first.probabilities.items()
                    ),
                    self._secondary.name: ",".join(
                        f"{k}={v:.3f}" for k, v in second.probabilities.items()
                    ),
                },
            )
        leader = max(first.probabilities.values(), default=0.0)
        if self._in_gray(leader):
            second = self._secondary.choose(question, rag_descriptions)
            return KbDist(
                probabilities=dict(second.probabilities),
                raw={"primary": first.raw, "secondary": second.raw},
                source=f"cascade:{self._secondary.name}",
                detail={
                    self._primary.name: f"leader={leader:.4f}",
                    self._secondary.name: ",".join(
                        f"{k}={v:.3f}" for k, v in second.probabilities.items()
                    ),
                },
            )
        return KbDist(
            probabilities=dict(first.probabilities),
            raw=first.raw,
            source=f"cascade:{self._primary.name}",
            detail={self._primary.name: f"leader={leader:.4f}"},
        )


class LegacyRoute:
    """Adapter: altes DecisionBackend (RouteDist inkl. 'none') -> DecisionRoute.

    D1: p_none = probabilities['none']; D2: uebrige Massen renormalisiert.
    """

    name = "legacy"

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def _dist(self, question: str, rag_descriptions: Mapping[str, str]) -> RouteDist:
        return self._backend.decide(question, rag_descriptions)

    def skip(
        self, question: str, rag_descriptions: Mapping[str, str] | None = None
    ) -> SkipDist:
        dist = self._dist(question, rag_descriptions or {})
        p_none = float(dist.probabilities.get("none", 0.0))
        return SkipDist(p_recall=1.0 - p_none, raw=dist.raw, source="legacy")

    def choose(
        self, question: str, rag_descriptions: Mapping[str, str]
    ) -> KbDist:
        dist = self._dist(question, rag_descriptions)
        probabilities = {
            key: float(dist.probabilities.get(key, 0.0)) for key in rag_descriptions
        }
        return KbDist(
            probabilities=_renormalize(probabilities),
            raw=dist.raw,
            source="legacy",
        )


# Beschreibung der No-Retrieval-Route bleibt importierbar (Kompatibilitaet).
__all__ = [
    "NO_RETRIEVAL_DESCRIPTION",
    "HybridRoute",
    "LayaRoute",
    "LegacyRoute",
    "SlmRoute",
    "build_kb_question",
    "build_skip_question",
]
