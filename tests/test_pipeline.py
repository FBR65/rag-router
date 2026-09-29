"""Tests fuer die Pipeline (P7): route_and_fetch end-to-end mit fakes."""

import pytest

from rag_router.backends.base import SearchHit
from rag_router.checking.base import CheckResult
from rag_router.config import RagBackendConfig, RagConfig, RouterConfig, Thresholds
from rag_router.decision.base import KbDist, RouteDist, SkipDist
from rag_router.pipeline import RagRouter, RouterResult


def make_config(
    rags: dict[str, RagConfig] | None = None,
    thresholds: Thresholds | None = None,
) -> RouterConfig:
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
        thresholds=thresholds if thresholds is not None else Thresholds(),
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


class SplitDecision:
    """Neuer Vertrag: D1/D2 getrennt (kein 'none' in choose)."""

    def __init__(self, p_recall: float, kb: dict[str, float]) -> None:
        self._p_recall = p_recall
        self._kb = kb

    def skip(self, question, rag_descriptions=None):
        return SkipDist(p_recall=self._p_recall, source="test")

    def choose(self, question, rag_descriptions):
        return KbDist(probabilities=dict(self._kb), source="test")


class TestSplitRoute:
    def test_skip_when_low_recall(self) -> None:
        router = RagRouter.injected(
            config=make_config(),
            decision=SplitDecision(0.1, {"policy": 0.9, "news": 0.1}),
            backends={"policy": FakeBackend("policy", ["p"])},
            checker=FakeChecker(0.9),
        )
        assert router.route("F").reason == "skip"
        assert router.route_and_fetch("F").final == "no_retrieval"

    def test_choose_without_none(self) -> None:
        router = RagRouter.injected(
            config=make_config(),
            decision=SplitDecision(0.9, {"policy": 0.8, "news": 0.2}),
            backends={"policy": FakeBackend("policy", ["p"])},
            checker=FakeChecker(0.9),
        )
        decision = router.route("F")
        assert decision.routes == ["policy"]
        # distribution enthaelt den none-Anteil (aus p_none) fuer Kompatibilitaet
        assert decision.distribution["none"] == pytest.approx(0.1)

    def test_fanout_from_split(self) -> None:
        router = RagRouter.injected(
            config=make_config(),
            decision=SplitDecision(0.9, {"policy": 0.30, "news": 0.28}),
            backends={"policy": FakeBackend("policy", ["p"])},
            checker=FakeChecker(0.9),
        )
        assert router.route("F").routes == ["policy", "news"]


class TestAutoCalibration:
    def test_auto_thresholds_from_profile(self) -> None:
        from rag_router.calibration import CalibrationProfile

        profile = CalibrationProfile(
            decision_route="laya",
            answer_backend="laya",
            skip=0.1,
            fanout=0.2,
            answer=0.5,
            cascade_lo=0.4,
            cascade_hi=0.6,
        )
        cfg = make_config(thresholds=Thresholds(skip=None, fanout=None, answer=None))
        router = RagRouter.injected(
            config=cfg,
            decision=SplitDecision(0.85, {"policy": 0.9, "news": 0.1}),
            backends={"policy": FakeBackend("policy", ["p"])},
            checker=FakeChecker(0.9),
            calibration=profile,
        )
        # skip=0.1 (auto aus Profil) -> p_none=0.15 >= 0.1 -> skip
        assert router.route("F").reason == "skip"
        assert router._calibration is profile

    def test_profile_to_dict_roundtrip_in_router(self) -> None:
        from rag_router.calibration import CalibrationProfile

        profile = CalibrationProfile(
            decision_route="slm",
            answer_backend="slm",
            skip=0.7,
            fanout=0.6,
            answer=0.4,
            cascade_lo=0.3,
            cascade_hi=0.8,
        )
        cfg = make_config(thresholds=Thresholds(skip=None, fanout=None, answer=None))
        router = RagRouter.injected(
            config=cfg,
            decision=SplitDecision(0.8, {"policy": 0.9, "news": 0.1}),
            backends={"policy": FakeBackend("policy", ["p"])},
            checker=FakeChecker(0.9),
            calibration=profile,
        )
        decision = router.route("F")
        # p_none = 0.2 < 0.7 -> suchen
        assert "none" not in decision.routes
