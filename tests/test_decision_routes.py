"""Tests fuer die Entscheidungswege (P10): D1/D2 je Weg getrennt.

Nur Logik/Fakes — echte Modelle laufen im Integrationstest. Deckt S1–S4, S6
der Spec (docs/spec-decision-stages.md).
"""

from types import SimpleNamespace

import pytest

from rag_router.decision.base import DecisionRoute, KbDist, RouteDist, SkipDist
from rag_router.decision.route import (
    HybridRoute,
    LayaRoute,
    LegacyRoute,
    SlmRoute,
)

RAG = {"policy": "Richtlinien", "news": "Nachrichten"}


# --------------------------------------------------------------------------- #
# Laya
# --------------------------------------------------------------------------- #
class FakeLayaPredict:
    def __init__(self, skip_p: float, kb: dict[str, float]) -> None:
        self._p_recall = skip_p
        self._kb = kb
        self.calls: list[dict] = []

    def __call__(self, state, questions, model=None, max_len=None, **kw):
        self.calls.append({"state": state, "questions": questions, "model": model})
        key = next(iter(questions))
        qdef = questions[key]
        if qdef["type"] == "noul":
            return {"answers": {key: {"type": "noul", "noul": 1.0 - self._p_recall}}}
        return {
            "answers": {
                key: {
                    "type": "choice",
                    "choice": max(self._kb, key=self._kb.get),
                    "probabilities": dict(self._kb),
                }
            }
        }


class TestLayaRoute:
    def test_split_uses_two_calls(self) -> None:
        predict = FakeLayaPredict(0.1, {"policy": 0.6, "news": 0.4})
        route = LayaRoute.from_predict(predict)
        skip = route.skip("Wie lange dauert die Rueckgabe?")
        kb = route.choose("Wie lange dauert die Rueckgabe?", RAG)
        assert isinstance(skip, SkipDist)
        assert isinstance(kb, KbDist)
        assert skip.p_recall == pytest.approx(0.1)
        assert skip.p_none == pytest.approx(0.9)
        assert len(predict.calls) == 2, "D1 und D2 = zwei getrennte predict-Calls"

    def test_skip_question_is_noul(self) -> None:
        predict = FakeLayaPredict(0.4, {"policy": 1.0})
        route = LayaRoute.from_predict(predict)
        route.skip("Frage")
        assert predict.calls[0]["questions"]["needs_docs"]["type"] == "noul"

    def test_choose_has_no_none(self) -> None:
        predict = FakeLayaPredict(0.4, {"policy": 0.7, "news": 0.3})
        route = LayaRoute.from_predict(predict)
        kb = route.choose("Frage", RAG)
        assert "none" not in kb.probabilities
        assert set(kb.probabilities) == {"policy", "news"}
        assert sum(kb.probabilities.values()) == pytest.approx(1.0)

    def test_choose_renormalizes(self) -> None:
        predict = FakeLayaPredict(0.4, {"policy": 0.6, "news": 0.2})
        route = LayaRoute.from_predict(predict)
        kb = route.choose("Frage", RAG)
        assert kb.probabilities["policy"] == pytest.approx(0.75)

    def test_name(self) -> None:
        assert LayaRoute.from_predict(FakeLayaPredict(0.0, {"policy": 1.0})).name == "laya"

    def test_protocol(self) -> None:
        assert isinstance(
            LayaRoute.from_predict(FakeLayaPredict(0.0, {"policy": 1.0})),
            DecisionRoute,
        )


# --------------------------------------------------------------------------- #
# SLM
# --------------------------------------------------------------------------- #
def make_response(top: list[tuple[str, float]] | None, content: str = "A"):
    logprobs = None
    if top is not None:
        logprobs = SimpleNamespace(
            content=[
                SimpleNamespace(
                    top_logprobs=[
                        SimpleNamespace(token=t, logprob=v) for t, v in top
                    ]
                )
            ]
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=logprobs, message=SimpleNamespace(content=content))]
    )


class FakeClient:
    def __init__(self, responses: list) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= len(self._responses):
            return self._responses[len(self.calls) - 1]
        return self._responses[-1]


class TestSlmRoute:
    def test_skip_choice_returns_p_none(self) -> None:
        # A=policy, B=news, C=none -> C stark -> p_none hoch
        client = FakeClient([make_response([("C", -0.1), ("A", -2.0), ("B", -2.5)])])
        route = SlmRoute(client, model="m")
        skip = route.skip("Frage", RAG)
        assert skip.p_none > 0.8
        assert skip.source == "logprobs"
        assert client.calls[0]["max_tokens"] == 1
        assert client.calls[0]["logprobs"] is True

    def test_skip_text_fallback(self) -> None:
        # Text-Fallback: Antwort ist 'C' = none-Label bei 2 KBs
        client = FakeClient([make_response(None, content="C")])
        route = SlmRoute(client, model="m")
        skip = route.skip("Frage", RAG)
        assert skip.p_none == pytest.approx(1.0)
        assert skip.source == "text"

    def test_choose_logprobs_over_kb(self) -> None:
        client = FakeClient([make_response([("A", -0.1), ("B", -3.0)])])
        route = SlmRoute(client, model="m")
        kb = route.choose("Frage", RAG)
        assert "none" not in kb.probabilities
        assert kb.probabilities["policy"] > kb.probabilities["news"]
        assert sum(kb.probabilities.values()) == pytest.approx(1.0)

    def test_choose_prompt_lists_all_kbs(self) -> None:
        client = FakeClient([make_response([("A", 0.0), ("B", -1.0)])])
        route = SlmRoute(client, model="m")
        route.choose("F", RAG)
        prompt = client.calls[0]["messages"][0]["content"]
        assert "Richtlinien" in prompt and "Nachrichten" in prompt

    def test_skip_prompt_lists_kbs_and_none(self) -> None:
        client = FakeClient([make_response([("A", -0.1), ("B", -2.0), ("C", -2.5)])])
        route = SlmRoute(client, model="m")
        route.skip("F", RAG)
        prompt = client.calls[0]["messages"][0]["content"]
        # KB-Beschreibungen UND none-Option stehen im Prompt
        assert "Richtlinien" in prompt and "Nachrichten" in prompt
        assert "Recherche" in prompt

    def test_split_uses_two_calls(self) -> None:
        client = FakeClient(
            [
                make_response([("A", -0.1), ("B", -2.0), ("C", -2.5)]),
                make_response([("A", -0.1), ("B", -2.0)]),
            ]
        )
        route = SlmRoute(client, model="m")
        route.skip("F", RAG)
        route.choose("F", RAG)
        assert len(client.calls) == 2

    def test_protocol(self) -> None:
        assert isinstance(SlmRoute(FakeClient([]), model="m"), DecisionRoute)


# --------------------------------------------------------------------------- #
# Hybrid
# --------------------------------------------------------------------------- #
class StubRoute:
    name = "stub"

    def __init__(self, name: str, skip_p: float, kb: dict[str, float]) -> None:
        self.name = name
        self._skip_p = skip_p
        self._kb = kb
        self.skip_calls = 0
        self.choose_calls = 0

    def skip(self, question, rag_descriptions=None):
        self.skip_calls += 1
        return SkipDist(p_recall=self._skip_p, source=self.name)

    def choose(self, question, rag_descriptions):
        self.choose_calls += 1
        return KbDist(probabilities=dict(self._kb), source=self.name)


class TestHybridCascade:
    def test_secondary_only_in_gray_zone(self) -> None:
        primary = StubRoute("p", skip_p=0.95, kb={"policy": 0.95, "news": 0.05})
        secondary = StubRoute("s", skip_p=0.1, kb={"policy": 0.1, "news": 0.9})
        route = HybridRoute(
            primary, secondary, strategy="cascade", cascade_lo=0.4, cascade_hi=0.6
        )
        skip = route.skip("F")
        assert skip.p_recall == pytest.approx(0.95)
        assert secondary.skip_calls == 0, "ausserhalb der Grauzone kein secondary"

    def test_gray_zone_bounds_inclusive(self) -> None:
        # exactly at lo and hi -> secondary greift (inklusive Grenzen)
        for value in (0.4, 0.6):
            primary = StubRoute("p", skip_p=value, kb={"policy": value, "news": 1 - value})
            secondary = StubRoute("s", skip_p=0.9, kb={"policy": 0.9, "news": 0.1})
            route = HybridRoute(
                primary, secondary, strategy="cascade", cascade_lo=0.4, cascade_hi=0.6
            )
            assert route.skip("F").p_recall == pytest.approx(0.9), (
                f"Grenze {value} muss in der Grauzone liegen (inklusiv)"
            )
            assert secondary.skip_calls == 1

    def test_secondary_takes_over_in_gray_zone(self) -> None:
        primary = StubRoute("p", skip_p=0.5, kb={"policy": 0.5, "news": 0.5})
        secondary = StubRoute("s", skip_p=0.9, kb={"policy": 0.9, "news": 0.1})
        route = HybridRoute(
            primary, secondary, strategy="cascade", cascade_lo=0.4, cascade_hi=0.6
        )
        skip = route.skip("F")
        assert skip.p_recall == pytest.approx(0.9)
        assert secondary.skip_calls == 1

    def test_choose_secondary_in_gray_zone(self) -> None:
        primary = StubRoute("p", skip_p=0.1, kb={"policy": 0.5, "news": 0.5})
        secondary = StubRoute("s", skip_p=0.1, kb={"policy": 0.9, "news": 0.1})
        route = HybridRoute(
            primary, secondary, strategy="cascade", cascade_lo=0.4, cascade_hi=0.6
        )
        kb = route.choose("F", RAG)
        # Fuehrung 0.5 liegt in [0.4,0.6] -> secondary
        assert kb.probabilities["policy"] == pytest.approx(0.9)


class TestHybridCommittee:
    def test_both_called(self) -> None:
        primary = StubRoute("p", skip_p=0.8, kb={"policy": 0.6, "news": 0.4})
        secondary = StubRoute("s", skip_p=0.4, kb={"policy": 0.2, "news": 0.8})
        route = HybridRoute(primary, secondary, strategy="committee", aggregate="mean")
        skip = route.skip("F")
        assert primary.skip_calls == 1 and secondary.skip_calls == 1
        assert skip.p_recall == pytest.approx(0.6)  # mean(0.8, 0.4)

    def test_mean(self) -> None:
        primary = StubRoute("p", skip_p=0.8, kb={"policy": 0.6, "news": 0.4})
        secondary = StubRoute("s", skip_p=0.4, kb={"policy": 0.2, "news": 0.8})
        route = HybridRoute(primary, secondary, strategy="committee", aggregate="mean")
        kb = route.choose("F", RAG)
        assert kb.probabilities["policy"] == pytest.approx(0.4)

    def test_max_and_renormalize(self) -> None:
        primary = StubRoute("p", skip_p=0.8, kb={"policy": 0.9, "news": 0.1})
        secondary = StubRoute("s", skip_p=0.4, kb={"policy": 0.3, "news": 0.7})
        route = HybridRoute(primary, secondary, strategy="committee", aggregate="max")
        kb = route.choose("F", RAG)
        # max je Key: policy 0.9, news 0.7 -> renormalisiert 0.9/1.6
        assert kb.probabilities["policy"] == pytest.approx(0.9 / 1.6)

    def test_detail_records_both(self) -> None:
        primary = StubRoute("p", skip_p=0.8, kb={"policy": 0.6, "news": 0.4})
        secondary = StubRoute("s", skip_p=0.4, kb={"policy": 0.2, "news": 0.8})
        route = HybridRoute(primary, secondary, strategy="committee", aggregate="mean")
        skip = route.skip("F")
        kb = route.choose("F", RAG)
        assert "p" in skip.detail and "s" in skip.detail
        assert "p" in kb.detail and "s" in kb.detail


# --------------------------------------------------------------------------- #
# Legacy-Adapter
# --------------------------------------------------------------------------- #
class FakeLegacyBackend:
    def decide(self, question, rag_descriptions):
        return RouteDist(probabilities={"policy": 0.6, "news": 0.2, "none": 0.2})


class TestLegacyRoute:
    def test_skip_from_none_mass(self) -> None:
        route = LegacyRoute(FakeLegacyBackend())
        assert route.skip("F").p_none == pytest.approx(0.2)

    def test_choose_renormalizes_without_none(self) -> None:
        route = LegacyRoute(FakeLegacyBackend())
        kb = route.choose("F", RAG)
        assert "none" not in kb.probabilities
        assert kb.probabilities["policy"] == pytest.approx(0.75)
        assert kb.probabilities["news"] == pytest.approx(0.25)

    def test_protocol(self) -> None:
        assert isinstance(LegacyRoute(FakeLegacyBackend()), DecisionRoute)
