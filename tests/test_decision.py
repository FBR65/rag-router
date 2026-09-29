"""Tests fuer die Decision-Backends (P3).

Die Backends wandeln eine Frage + RAG-Beschreibungen in einen
Wahrscheinlichkeitsvektor je Route (inklusive "none"). Hier nur Logik —
die echten Modelle kommen im Integration-Test (Marker integration).
"""

from types import SimpleNamespace

from rag_router.decision.base import DecisionBackend, RouteDist
from rag_router.decision.laya import LayaDecisionBackend
from rag_router.decision.llm import LlmDecisionBackend

RAG_DESCRIPTIONS = {
    "policy": "Richtlinien und Hausregeln",
    "news": "Nachrichten und Ereignisse",
}


def fake_laya_payload(probs: dict[str, float]) -> dict:
    return {
        "answers": {
            "route": {
                "type": "choice",
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
                "confidence": 0.5,
            }
        }
    }


class TestLayaBackend:
    def test_builds_criteria_none_last(self) -> None:
        q = LayaDecisionBackend.build_question(RAG_DESCRIPTIONS)
        criteria = q["route"]["criteria"]
        assert list(criteria) == ["policy", "news", "none"]
        assert q["route"]["type"] == "choice"
        assert criteria["none"]
        assert q["route"]["instructions"]

    def test_decide_maps_probabilities(self) -> None:
        calls: list[tuple] = []

        def predict(state, questions, model=None, max_len=None, **kw):
            calls.append((state, model, max_len))
            return fake_laya_payload({"policy": 0.38, "news": 0.19, "none": 0.43})

        backend = LayaDecisionBackend.from_predict(predict, max_len=1024)
        dist = backend.decide("Wie viele Tage Urlaub?", RAG_DESCRIPTIONS)

        assert isinstance(dist, RouteDist)
        assert dist.probabilities == {"policy": 0.38, "news": 0.19, "none": 0.43}
        assert dist.raw["answers"]["route"]["choice"]
        # state=Frage, explizites Sprach-Modell, max_len aus Konfig
        assert calls[0][0] == "Wie viele Tage Urlaub?"
        assert calls[0][2] == 1024

    def test_missing_route_probability_is_zero(self) -> None:
        def predict(state, questions, **kw):
            return fake_laya_payload({"policy": 0.9, "news": 0.1})

        backend = LayaDecisionBackend.from_predict(predict)
        dist = backend.decide("F", RAG_DESCRIPTIONS)
        assert dist.probabilities["none"] == 0.0
        assert dist.probabilities["policy"] == 0.9

    def test_implements_protocol(self) -> None:
        def predict(state, questions, **kw):
            return fake_laya_payload({"none": 1.0})

        backend = LayaDecisionBackend.from_predict(predict)
        assert isinstance(backend, DecisionBackend)

    def test_model_override_passed_through(self) -> None:
        seen: dict = {}

        def predict(state, questions, model=None, **kw):
            seen["model"] = model
            return fake_laya_payload({"none": 1.0})

        backend = LayaDecisionBackend.from_predict(predict, model="english")
        backend.decide("F", RAG_DESCRIPTIONS)
        assert seen["model"] == "english"


def make_llm_response(top_logprobs: list[tuple[str, float]] | None, content: str = "A"):
    """OpenAI-ChatCompletion-Nachbau (nur die Felder, die wir lesen)."""
    logprobs = None
    if top_logprobs is not None:
        logprobs = SimpleNamespace(
            content=[
                SimpleNamespace(
                    top_logprobs=[
                        SimpleNamespace(token=tok, logprob=lp)
                        for tok, lp in top_logprobs
                    ]
                )
            ]
        )
    choice = SimpleNamespace(
        logprobs=logprobs, message=SimpleNamespace(content=content)
    )
    return SimpleNamespace(choices=[choice])


class FakeLlmClient:
    def __init__(self, response, calls: list | None = None) -> None:
        self._response = response
        self.calls = calls if calls is not None else []

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class TestLlmBackend:
    def test_builds_labels_and_mapping(self) -> None:
        labels, criteria = LlmDecisionBackend.build_options(RAG_DESCRIPTIONS)
        assert labels == {"policy": "A", "news": "B", "none": "C"}
        assert list(criteria) == ["policy", "news", "none"]

    def test_decide_from_top_logprobs(self) -> None:
        calls: list = []
        client = FakeLlmClient(
            make_llm_response([("A", -0.1), ("B", -2.3), ("C", -3.5)]), calls
        )
        backend = LlmDecisionBackend.from_client(client, model="test-model")
        dist = backend.decide("Wie viele Tage Urlaub?", RAG_DESCRIPTIONS)

        import math

        a, b, c = -0.1, -2.3, -3.5
        denom = math.exp(0.0) + math.exp(b - a) + math.exp(c - a)
        assert dist.probabilities["policy"] == pytest.approx(1.0 / denom, abs=1e-3)
        assert dist.probabilities["news"] == pytest.approx(
            math.exp(b - a) / denom, abs=1e-3
        )
        assert dist.probabilities["none"] == pytest.approx(
            math.exp(c - a) / denom, abs=1e-3
        )
        assert calls[0]["model"] == "test-model"
        assert calls[0]["max_tokens"] == 1
        assert calls[0]["logprobs"] is True

    def test_decide_without_logprobs_one_hot(self) -> None:
        client = FakeLlmClient(make_llm_response(None, content="B"))
        backend = LlmDecisionBackend.from_client(client, model="m")
        dist = backend.decide("F", RAG_DESCRIPTIONS)
        assert dist.probabilities == {"policy": 0.0, "news": 1.0, "none": 0.0}
        assert dist.probabilities_source == "text"

    def test_unparseable_label_falls_back_to_first_rag(self) -> None:
        client = FakeLlmClient(make_llm_response(None, content="42"))
        backend = LlmDecisionBackend.from_client(client, model="m")
        dist = backend.decide("F", RAG_DESCRIPTIONS)
        assert dist.probabilities["policy"] == 1.0
        assert dist.probabilities["none"] == 0.0

    def test_too_many_routes_rejected(self) -> None:
        many = {f"rag_{i:02d}": "x" for i in range(26)}  # 26 RAGs + none = 27
        with pytest.raises(ValueError, match="27"):
            LlmDecisionBackend.build_options(many)

    def test_prompt_contains_question_and_options(self) -> None:
        calls: list = []
        client = FakeLlmClient(
            make_llm_response([("A", 0.0), ("B", -1.0), ("C", -2.0)]), calls
        )
        backend = LlmDecisionBackend.from_client(client, model="m")
        backend.decide("Wie viele Tage?", RAG_DESCRIPTIONS)
        prompt = calls[0]["messages"][0]["content"]
        assert "Wie viele Tage?" in prompt
        assert "A." in prompt and "B." in prompt and "C." in prompt


import pytest

from rag_router.decision.base import RouteDist as _RD  # noqa: F401
