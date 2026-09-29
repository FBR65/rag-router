"""LLM (OpenAI-kompatibel) als Answer-Checker (P6).

Yes/No-Frage mit max_tokens=1 und logprob-Fergleich A=yes/B=no;
Softmax auf beide Labels -> p(answered).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from rag_router.checking.base import CheckResult

INSTRUCTIONS = "Beantwortet der folgende Kontext die gestellte Frage inhaltlich?"

_LABELS = ("A", "B")  # A = beantwortet (ja), B = beantwortet nicht (nein)
_LABEL_PATTERN = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")


def build_prompt(question: str, passages: Sequence[str]) -> str:
    context = "\n\n".join(
        f"[{index}] {passage}" for index, passage in enumerate(passages, start=1)
    )
    return (
        f"{INSTRUCTIONS}\n\n"
        f"A = Ja, der Kontext enthält die Antwort.\n"
        f"B = Nein, der Kontext enthält die Antwort nicht.\n\n"
        f"Frage: {question}\n\nKontext:\n{context}\n\n"
        "Antworte ausschließlich mit dem Buchstaben."
    )


class LlmAnswerChecker:
    """AnswerChecker ueber einen OpenAI-kompatiblen Chat-Endpunkt."""

    def __init__(self, client: Any, *, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_client(cls, client: Any, *, model: str) -> LlmAnswerChecker:
        return cls(client, model=model)

    @classmethod
    def from_settings(
        cls, *, base_url: str, api_key: str, model: str
    ) -> LlmAnswerChecker:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("llm checker benoetigt das Paket 'openai'") from error
        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        return cls(client, model=model)

    def check(self, question: str, passages: Sequence[str]) -> CheckResult:
        prompt = build_prompt(question, passages)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
        choice = response.choices[0]
        logprobs = choice.logprobs
        score_map: dict[str, float] = {}
        if logprobs is not None and logprobs.content:
            for entry in logprobs.content[0].top_logprobs or ():
                token = entry.token.strip()
                if token in _LABELS:
                    score_map[token] = max(
                        score_map.get(token, entry.logprob), entry.logprob
                    )
        if score_map:
            best = max(score_map.values())
            exps = {
                label: math.exp(score_map.get(label, -math.inf) - best)
                for label in _LABELS
            }
            total = sum(exps.values())
            return CheckResult(
                p_answered=float(exps["A"] / total),
                raw=response,
                source="logprobs",
            )
        content = (choice.message.content or "").strip()
        match = _LABEL_PATTERN.search(content)
        p_answered = 1.0 if match and match.group(1) == "A" else 0.0
        return CheckResult(p_answered=p_answered, raw=response, source="text")
