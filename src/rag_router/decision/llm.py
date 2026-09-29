"""OpenAI-kompatibles Decision-Backend (P3).

Muster aus dem LLM-Router: der Endpunkt bekommt einen Prompt mit A/B/C-...
Optionen und liefert max_tokens=1 mit logprobs; die Label-Logprobs werden
per Softmax zu Wahrscheinlichkeiten normalisiert. Ohne logprobs-Support
(one-hot aus dem Antworttext, gekennzeichnet mit probabilities_source=text).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Protocol

from rag_router.decision.base import (
    NO_RETRIEVAL_DESCRIPTION,
    NO_RETRIEVAL_KEY,
    RouteDist,
)

MAX_LABELS = 25  # A..Y
_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LABEL_PATTERN = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")

INSTRUCTIONS = "Wähle die passende Wissensbasis."


class ChatCompletionsClient(Protocol):
    """Der minimale OpenAI-Client-Surface, das dieses Backend liest."""

    def __init__(self, **kwargs: Any) -> None: ...  # noqa: E704


def build_options(
    rag_descriptions: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """RAG-Keys -> A/B/C-Labels + Kriterien-Map (Reihenfolge = Eingabe)."""
    if len(rag_descriptions) + 1 > MAX_LABELS:
        raise ValueError(
            f"zu viele Routen ({len(rag_descriptions) + 1}); Max: {MAX_LABELS}"
        )
    labels: dict[str, str] = {}
    criteria: dict[str, str] = {}
    items = list(rag_descriptions.items())
    for index, (key, description) in enumerate(items):
        labels[key] = _LABELS[index]
        criteria[key] = description
    labels[NO_RETRIEVAL_KEY] = _LABELS[len(items)]
    criteria[NO_RETRIEVAL_KEY] = NO_RETRIEVAL_DESCRIPTION
    return labels, criteria


def build_prompt(question: str, criteria: Mapping[str, str]) -> str:
    options = "\n".join(
        f"{_LABELS[index]}. {criteria_key}"
        for index, criteria_key in enumerate(criteria)
    )
    return (
        f"{INSTRUCTIONS} Wähle genau eine Option.\n\n"
        f"Frage: {question}\n\n{options}\n\n"
        "Antworte ausschließlich mit dem Buchstaben der Option."
    )


class LlmDecisionBackend:
    """DecisionBackend ueber einen OpenAI-kompatiblen Chat-Endpunkt."""

    build_options = staticmethod(build_options)
    build_prompt = staticmethod(build_prompt)

    def __init__(
        self,
        client: Any,
        *,
        model: str,
    ) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_client(cls, client: Any, *, model: str) -> LlmDecisionBackend:
        return cls(client, model=model)

    @classmethod
    def from_settings(
        cls, *, base_url: str, api_key: str, model: str
    ) -> LlmDecisionBackend:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("llm backend benoetigt das Paket 'openai'") from error
        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        return cls(client, model=model)

    def decide(self, question: str, rag_descriptions: Mapping[str, str]) -> RouteDist:
        labels, criteria = build_options(rag_descriptions)
        prompt = build_prompt(question, criteria)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
        choice = response.choices[0]
        logprobs = choice.logprobs
        score_map: dict[str, float] | None = None
        if logprobs is not None and logprobs.content:
            score_map = {}
            for entry in logprobs.content[0].top_logprobs or ():
                token = entry.token.strip()
                if token in labels.values():
                    score_map[token] = max(
                        score_map.get(token, entry.logprob), entry.logprob
                    )
        if score_map:
            return RouteDist(
                probabilities=self._from_scores(score_map, labels),
                raw=response,
                probabilities_source="logprobs",
            )
        # Endpunkt ohne (brauchbare) logprobs:
        content = (choice.message.content or "").strip()
        return self._one_hot_from_text(content, labels, response)

    def _from_scores(
        self, score_map: Mapping[str, float], labels: Mapping[str, str]
    ) -> dict[str, float]:
        label_by_route = {v: k for k, v in labels.items()}
        best = max(score_map.values())
        total = sum(math.exp(s - best) for s in score_map.values())
        probabilities = {
            label_by_route[label]: float(
                math.exp(score_map.get(label, -math.inf) - best) / total
            )
            for label in label_by_route
        }
        return probabilities

    def _one_hot_from_text(
        self, content: str, labels: Mapping[str, str], raw: Any
    ) -> RouteDist:
        match = _LABEL_PATTERN.search(content or "")
        label = match.group(1) if match else None
        probabilities = {key: 0.0 for key in labels}
        if label is None:
            # fail closed auf die erste RAG-Route (Artikel: skip ist teurer
            # als Suche; hier: falsche KB-Wahl ist billiger als none)
            first_rag = next(iter(labels))
            probabilities[first_rag] = 1.0
        else:
            route = next(k for k, v in labels.items() if v == label)
            probabilities[route] = 1.0
        return RouteDist(
            probabilities=probabilities, raw=raw, probabilities_source="text"
        )
