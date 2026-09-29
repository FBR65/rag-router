"""Laya als Decision-Backend (typed-decision model, P3).

Ein `choice`-Forward-Pass ueber alle Routen gleichzeitig; kalibrierte
Wahrscheinlichkeiten aus `answers[qid]["probabilities"]` (Payload-Form im
Layapaket verifiziert: choice -> dict optionkey -> float).
Importiert das Layapaket nur hier — wie im LLM-Router bewusst optional.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from rag_router.decision.base import (
    NO_RETRIEVAL_DESCRIPTION,
    NO_RETRIEVAL_KEY,
    RouteDist,
)

QUESTION_KEY = "route"
ROUTING_INSTRUCTIONS = "Welche Wissensbasis sollte diese Frage beantworten?"
PredictFn = Callable[..., Mapping[str, Any]]


def build_question(rag_descriptions: Mapping[str, str]) -> dict[str, Any]:
    """Typed-Question fuer Laya: Kriterien = RAG-Keys + 'none' (zuletzt)."""
    criteria: dict[str, str] = {}
    for key, description in rag_descriptions.items():
        criteria[key] = description
    criteria[NO_RETRIEVAL_KEY] = NO_RETRIEVAL_DESCRIPTION
    return {
        QUESTION_KEY: {
            "type": "choice",
            "instructions": ROUTING_INSTRUCTIONS,
            "criteria": criteria,
        }
    }


class LayaDecisionBackend:
    """DecisionBackend auf Basis eines Laya-Agents/Router-`predict`."""

    build_question = staticmethod(build_question)

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
    ) -> LayaDecisionBackend:
        try:
            from laya import Router
        except ImportError as error:
            raise RuntimeError(
                "laya backend benoetigt das Paket 'laya': uv add laya"
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
    ) -> LayaDecisionBackend:
        """Direkt aus einer predict-Funktion (Tests, Injektion)."""
        return cls(predict=predict, model=model, max_len=max_len)

    def decide(self, question: str, rag_descriptions: Mapping[str, str]) -> RouteDist:
        payload = self._predict(
            question,
            build_question(rag_descriptions),
            model=self._model,
            max_len=self._max_len,
        )
        answers = payload.get("answers") or {}
        answer = answers.get(QUESTION_KEY) or {}
        raw_probs = answer.get("probabilities") or {}
        probabilities = {
            key: float(raw_probs.get(key, 0.0))
            for key in (*rag_descriptions, NO_RETRIEVAL_KEY)
        }
        return RouteDist(probabilities=probabilities, raw=payload)
