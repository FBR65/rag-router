"""Tests fuer den Answer-Check (P6): beantworten Passagen die Frage?"""

from types import SimpleNamespace

import pytest

from rag_router.checking.base import AnswerChecker, CheckResult
from rag_router.checking.laya import LayaAnswerChecker
from rag_router.checking.llm import LlmAnswerChecker

QUESTION = "Wie lange dauert die Rueckgabe?"
PASSAGES = [
    "Rueckgabe: Artikel koennen 14 Tage unbenutzt zurueckgegeben werden.",
    "Urlaubsantrag: 30 Tage Restverfall im Maerz.",
]


class TestLayaChecker:
    def test_protocol_implemented(self) -> None:
        checker = LayaAnswerChecker.from_predict(lambda s, q, **kw: {})
        assert isinstance(checker, AnswerChecker)

    def test_noul_question_shape(self) -> None:
        q = LayaAnswerChecker.build_question()
        assert list(q) == ["answered"]
        assert q["answered"]["type"] == "noul"

    def test_state_contains_question_and_passages(self) -> None:
        seen: dict = {}

        def predict(state, questions, model=None, max_len=None, **kw):
            seen["state"] = state
            return {"answers": {"answered": {"type": "noul", "noul": 0.9}}}

        checker = LayaAnswerChecker.from_predict(predict)
        result = checker.check(QUESTION, PASSAGES)

        assert isinstance(result, CheckResult)
        assert result.p_answered == pytest.approx(0.9, abs=1e-6)
        assert QUESTION in seen["state"]
        for passage in PASSAGES:
            assert passage in seen["state"]
        assert result.raw["answers"]["answered"]["noul"] == 0.9

    def test_missing_answer_defaults_to_zero(self) -> None:
        checker = LayaAnswerChecker.from_predict(lambda s, q, **kw: {"answers": {}})
        result = checker.check(QUESTION, PASSAGES)
        assert result.p_answered == 0.0
        assert result.error is not None and "keine" in result.error


class FakeLlmClient:
    def __init__(self, response) -> None:
        self._response = response
        self.calls: list = []

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _resp(logprobs_pairs, content="A"):
    lp = (
        SimpleNamespace(
            content=[
                SimpleNamespace(
                    top_logprobs=[
                        SimpleNamespace(token=t, logprob=v) for t, v in logprobs_pairs
                    ]
                )
            ]
        )
        if logprobs_pairs is not None
        else None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=lp, message=SimpleNamespace(content=content))]
    )


class TestLlmChecker:
    def test_logprobs_yes_no(self) -> None:
        import math

        client = FakeLlmClient(_resp([("A", -0.2), ("B", -1.7)]))
        checker = LlmAnswerChecker.from_client(client, model="m")
        result = checker.check(QUESTION, PASSAGES)
        a, b = -0.2, -1.7
        denom = 1.0 + math.exp(b - a)
        assert result.p_answered == pytest.approx(1.0 / denom, abs=1e-3)
        assert result.source == "logprobs"

    def test_text_fallback(self) -> None:
        client = FakeLlmClient(_resp(None, content="B"))
        checker = LlmAnswerChecker.from_client(client, model="m")
        result = checker.check(QUESTION, PASSAGES)
        assert result.p_answered == 0.0
        assert result.source == "text"

    def test_prompt_mentions_question_and_passages(self) -> None:
        client = FakeLlmClient(_resp([("A", 0.0), ("B", -5.0)]))
        checker = LlmAnswerChecker.from_client(client, model="m")
        checker.check(QUESTION, PASSAGES)
        prompt = client.calls[0]["messages"][0]["content"]
        assert QUESTION in prompt
        assert any(p in prompt for p in PASSAGES)
