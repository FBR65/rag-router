"""Laya als Answer-Checker (P6): noul-Frage -> p(true) = p(answered).

Payload-Form im Layapaket verifiziert: noul -> answers[qid]["noul"] = P(true).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from rag_router.checking.base import CheckResult

QUESTION_KEY = "answered"
CHECK_INSTRUCTIONS = "Beantworten die folgenden Passagen die Frage inhaltlich?"
PredictFn = Callable[..., Any]

ANSWERED_CRITERIA = {
    "false": "Die Passagen beantworten die Frage nicht.",
    "true": "Die Passagen enthalten die Antwort.",
}


def build_question() -> dict[str, Any]:
    """Noul-Frage: eine Option, p(true) = beantwortet."""
    return {
        QUESTION_KEY: {
            "type": "noul",
            "instructions": CHECK_INSTRUCTIONS,
            "criteria": ANSWERED_CRITERIA,
        }
    }


def build_state(question: str, passages: Sequence[str]) -> str:
    """Frage + Passagen als ein State-String."""
    blocks = [f"Frage: {question}"]
    for index, passage in enumerate(passages, start=1):
        blocks.append(f"Passage {index}: {passage}")
    return "\n\n".join(blocks)


class LayaAnswerChecker:
    """AnswerChecker auf Basis eines Laya-`predict`."""

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
    ) -> LayaAnswerChecker:
        try:
            from laya import Router
        except ImportError as error:
            raise RuntimeError(
                "laya checker benoetigt das Paket 'laya': uv add laya"
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
    ) -> LayaAnswerChecker:
        return cls(predict=predict, model=model, max_len=max_len)

    def check(self, question: str, passages: Sequence[str]) -> CheckResult:
        state = build_state(question, passages)
        payload = self._predict(
            state,
            build_question(),
            model=self._model,
            max_len=self._max_len,
        )
        answers = payload.get("answers") or {}
        answer = answers.get(QUESTION_KEY) or {}
        noul = answer.get("noul")
        if not isinstance(noul, (int, float)):
            return CheckResult(
                p_answered=0.0,
                raw=payload,
                error=f"Laya lieferte keine noul-Wahrscheinlichkeit ({QUESTION_KEY})",
                source="missing",
            )
        return CheckResult(p_answered=float(noul), raw=payload)


class LayaAnswerCheckerProtocolFactory:
    """Nur der Verstaendlichkeit halber: noul-Wert ist P(true)."""
