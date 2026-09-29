"""Tests fuer die Pipeline (P7): route_and_fetch end-to-end mit fakes."""

import pytest

from rag_router.backends.base import SearchHit
from rag_router.checking.base import CheckResult
from rag_router.config import RagBackendConfig, RagConfig, RouterConfig, Thresholds
from rag_router.decision.base import RouteDist
from rag_router.pipeline import RagRouter, RouterResult


def make_config(rags: dict[str, RagConfig] | None = None) -> RouterConfig:
    if rags is None:
        rags = {
            "policy": RagConfig(
                key="policy",
                description="Richtlinien",
                backend=RagBackendConfig(type="lancedb", db="./d", table="t1"),
                retriever="hybrid",
                top_k=3,
                rerank=True,
            ),
            "news": RagConfig(
                key="news",
                description="Nachrichten",
                backend=RagBackendConfig(type="lancedb", db="./d", table="t2"),
                retriever="dense",
                top_k=5,
                rerank=False,
            ),
        }
    return RouterConfig(
        backend="laya",
        laya=None,  # type: ignore[arg-type]
        llm=None,
        thresholds=Thresholds(),
        defaults=None,  # type: ignore[arg-type]
        rags=rags,
    )


class FakeDecision:
    def __init__(self, routes: list[str], probabilities: dict[str, float]) -> None:
        self._routes = routes
        self._probabilities = probabilities

    def decide(self, question, rag_descriptions):
        return RouteDist(probabilities=self._probabilities)


class FakeBackend:
    """Ein Fake-RAG: liefert feste Hits je Mode."""

    def __init__(self, rag_key: str, hits: list[str]) -> None:
        self._rag_key = rag_key
        self._hits = hits
        self.searched: list[str] = []

    def search(self, question, top_k, modes=None):
        self.searched.append(question)
        return [
            SearchHit(
                rag_key=self._rag_key,
                doc_id=f"{self._rag_key}-{i}",
                text=t,
                score=0.9 - i * 0.1,
                scores={},
            )
            for i, t in enumerate(self._hits)
        ]


class FakeChecker:
    def __init__(self, p_answered: float) -> None:
        self._p = p_answered
        self.calls: list[tuple[str, list[str]]] = []

    def check(self, question, passages):
        self.calls.append((question, list(passages)))
        return CheckResult(p_answered=self._p)


class TestNoRetrieval:
    def test_none_route_skips_retrieval(self) -> None:
        backend = FakeBackend("policy", ["x"])
        checker = FakeChecker(0.9)
        router = RagRouter.injected(
            config=make_config(),
            decision=FakeDecision(["none"], {"policy": 0.2, "none": 0.8}),
            backends={"policy": backend, "news": FakeBackend("news", [])},
            checker=checker,
        )
        result = router.route_and_fetch("Danke, mir reicht's!")
        assert isinstance(result, RouterResult)
        assert result.decision.routes == ["none"]
        assert result.hits == []
        assert result.check is None
        assert result.final == "no_retrieval"
        assert backend.searched == [], "bei skip wird nicht gesucht"


class TestSingleRoute:
    def test_answered(self) -> None:
        backend = FakeBackend("policy", ["Policy-Passage A", "Policy-Passage B"])
        checker = FakeChecker(0.92)
        router = RagRouter.injected(
            config=make_config(),
            decision=FakeDecision(
                ["policy"], {"policy": 0.6, "news": 0.2, "none": 0.2}
            ),
            backends={"policy": backend},
            checker=checker,
        )
        result = router.route_and_fetch("Wie lange dauert die Rueckgabe?")
        assert result.decision.routes == ["policy"]
        assert len(result.hits) == 2
        assert result.check is not None and result.check.p_answered == pytest.approx(
            0.92
        )
        assert result.final == "answered"
        assert backend.searched == ["Wie lange dauert die Rueckgabe?"]

    def test_not_found_below_cutoff(self) -> None:
        router = RagRouter.injected(
            config=make_config(),
            decision=FakeDecision(["policy"], {"policy": 0.6, "none": 0.4}),
            backends={"policy": FakeBackend("policy", ["p1"])},
            checker=FakeChecker(0.31),
        )
        result = router.route_and_fetch("F")
        assert result.final == "not_found"


class TestFanout:
    def test_fanout_checks_both_and_takes_best(self) -> None:
        backend_p = FakeBackend("policy", ["policy passage"])
        backend_n = FakeBackend("news", ["news passage"])
        checker = FakeChecker(0.8)
        router = RagRouter.injected(
            config=make_config(),
            decision=FakeDecision(
                ["policy", "news"], {"policy": 0.28, "news": 0.24, "none": 0.48}
            ),
            backends={"policy": backend_p, "news": backend_n},
            checker=checker,
        )
        result = router.route_and_fetch("F")
        assert result.decision.routes == ["policy", "news"]
        assert len(result.hits) == 2
        assert len(checker.calls) == 2, "ein Check je gesuchtem RAG"
        assert result.final == "answered"
        # RAG-Key je Hit gesetzt, nicht der leere Backend-Key
        assert {h.rag_key for h in result.hits} == {"policy", "news"}


class TestRespectsTopK:
    def test_backend_gets_rag_top_k(self) -> None:
        class CountingBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__("policy", ["a", "b"])
                self.last_top_k = None

            def search(self, question, top_k, modes=None):
                self.last_top_k = top_k
                return super().search(question, top_k, modes)

        counting = CountingBackend()
        router = RagRouter.injected(
            config=make_config(),
            decision=FakeDecision(["policy"], {"policy": 0.7, "none": 0.3}),
            backends={"policy": counting},
            checker=FakeChecker(0.8),
        )
        router.route_and_fetch("F")
        assert counting.last_top_k == 3  # top_k aus RagConfig

        router.route_and_fetch("F", top_k=1)
        assert counting.last_top_k == 1  # Override
