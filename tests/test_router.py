"""Tests fuer die Router-Kernlogik (P4): skip, renormalize, fan-out."""

import pytest

from rag_router.decision.base import KbDist, RouteDist, SkipDist
from rag_router.router import DecisionThresholds, RouterDecisionEngine

DIST_3_1 = {"policy": 0.38, "news": 0.19, "none": 0.43}


def make_backend(probs: dict[str, float]) -> object:
    class FakeBackend:
        def decide(self, question, rag_descriptions):
            return RouteDist(probabilities=dict(probs))

    return FakeBackend()


def make_split(p_recall: float, kb: dict[str, float]) -> object:
    class FakeSplit:
        def skip(self, question, rag_descriptions=None):
            return SkipDist(p_recall=p_recall)

        def choose(self, question, rag_descriptions):
            return KbDist(probabilities=dict(kb))

    return FakeSplit()


class TestSkip:
    def test_skip_at_threshold(self) -> None:
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.25, "none": 0.75}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("Wie ist das Wetter?")
        assert decision.routes == ["none"]
        assert decision.reason == "skip"
        assert decision.distribution["none"] == 0.75

    def test_no_skip_below_threshold(self) -> None:
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.38, "news": 0.19, "none": 0.59}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("Wie viele Tage Urlaub?")
        assert "none" not in decision.routes
        assert decision.reason in {"clear_leader", "fanout"}

    def test_skip_boundary_inclusive(self) -> None:
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.25, "none": 0.60}),
            thresholds=DecisionThresholds(),
        )
        assert engine.route("x").routes == ["none"]


class TestSplitRouteSkipBoundary:
    def test_split_skip_boundary_inclusive(self) -> None:
        engine = RouterDecisionEngine(
            make_split(p_recall=0.40, kb={"policy": 1.0}),
            thresholds=DecisionThresholds(),
        )
        # p_none = 0.60 == skip -> skip (inklusiv)
        assert engine.route("x").routes == ["none"]

    def test_split_just_below_skips_not(self) -> None:
        engine = RouterDecisionEngine(
            make_split(p_recall=0.41, kb={"policy": 1.0}),
            thresholds=DecisionThresholds(),
        )
        assert engine.route("x").routes == ["policy"]


class TestFanoutBoundary:
    def test_fanout_boundary_exact_is_clear_leader(self) -> None:
        # renormalisiert exakt 0.55 -> NICHT kleiner -> clear_leader
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.55, "news": 0.45, "none": 0.0}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        assert decision.reason == "clear_leader"
        assert decision.routes == ["policy"]

    def test_split_fanout_boundary_exact_is_clear_leader(self) -> None:
        # Split-Pfad: KB 0.55/0.45 -> renormalisiert 0.55 -> clear_leader
        engine = RouterDecisionEngine(
            make_split(p_recall=0.9, kb={"policy": 0.55, "news": 0.45}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        assert decision.reason == "clear_leader"
        assert decision.routes == ["policy"]

    def test_split_fanout_below_boundary(self) -> None:
        engine = RouterDecisionEngine(
            make_split(p_recall=0.9, kb={"policy": 0.54, "news": 0.46}),
            thresholds=DecisionThresholds(),
        )
        assert engine.route("F").routes == ["policy", "news"]


class TestFanout:
    def test_fanout_below_threshold(self) -> None:
        # p(none)=0.45 -> search; KB-renormalisiert: policy .545, news .455
        # -> Fuehrung < 0.55 -> Fan-out auf Top-2
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.30, "news": 0.25, "none": 0.45}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("Wie viele Tage Urlaub?")
        assert decision.routes == ["policy", "news"]
        assert decision.reason == "fanout"

    def test_clear_leader_above_threshold(self) -> None:
        # KB-renormalisiert: 0.44/(0.44+0.35) = 0.5570 > 0.55 -> clear leader
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.44, "news": 0.35, "none": 0.21}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        assert decision.routes == ["policy"]
        assert decision.reason == "clear_leader"

    def test_fanout_boundary_below(self) -> None:
        # renormalisiert: 0.5499 < 0.55 -> fanout
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.45, "news": 0.37, "none": 0.18}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        assert decision.routes == ["policy", "news"]


class TestRenormalize:
    def test_renormalization_ignores_none(self) -> None:
        engine = RouterDecisionEngine(
            make_backend(DIST_3_1),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        # (0.38+0.19)/(0.38+0.19) = 1.0 — none fliegt aus der Wahl raus
        assert decision.routes[0] == "policy"

    def test_single_rag_no_fanout(self) -> None:
        # 1 RAG (p_none=0.59 knapp unter Skip): kein Fanout moeglich
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.3, "none": 0.59}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        assert decision.routes == ["policy"]


class TestTies:
    def test_tie_broken_by_yaml_order(self) -> None:
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.4, "news": 0.4, "none": 0.2}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        # Beide gleich stark -> fanout mit beiden, Reihenfolge = Eingabe
        assert decision.routes == ["policy", "news"]
        assert decision.reason == "fanout"

    def test_tie_single_option_favors_first(self) -> None:
        # 1 RAG, tie zwischen rag und none unter Skip-Schwelle
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.5, "none": 0.5}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        assert decision.routes == ["policy"]


class TestDistribution:
    def test_distribution_passthrough(self) -> None:
        engine = RouterDecisionEngine(
            make_backend(DIST_3_1), thresholds=DecisionThresholds()
        )
        decision = engine.route("F")
        assert decision.distribution == DIST_3_1

    def test_reason_documented(self) -> None:
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.9, "none": 0.1}),
            thresholds=DecisionThresholds(),
        )
        decision = engine.route("F")
        assert decision.reason == "clear_leader"


class TestCustomThresholds:
    def test_custom_skip(self) -> None:
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.55, "none": 0.45}),
            thresholds=DecisionThresholds(skip=0.40),
        )
        assert engine.route("F").routes == ["none"]

    def test_custom_fanout(self) -> None:
        engine = RouterDecisionEngine(
            make_backend({"policy": 0.45, "news": 0.369, "none": 0.181}),
            thresholds=DecisionThresholds(fanout=0.50),
        )
        decision = engine.route("F")
        assert decision.routes == ["policy"]  # 0.55 >= 0.50


def test_negative_probabilities_rejected() -> None:
    engine = RouterDecisionEngine(
        make_backend({"policy": 1.5, "none": -0.5}),
        thresholds=DecisionThresholds(),
    )
    with pytest.raises(ValueError, match="negativ"):
        engine.route("F")


def test_empty_probabilities_rejected() -> None:
    engine = RouterDecisionEngine(make_backend({}), thresholds=DecisionThresholds())
    with pytest.raises(ValueError, match="leer"):
        engine.route("F")
