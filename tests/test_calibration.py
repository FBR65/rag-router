"""Tests fuer die Kalibrierung (P11/Auto-Werte)."""

import pytest

from rag_router.calibration import (
    FALLBACK_ANSWER,
    FALLBACK_FANOUT,
    FALLBACK_SKIP,
    CalibrationProfile,
    CalibrationSample,
    choose_best_answer_backend,
    choose_best_route,
    derive,
    fallback_profile,
    route_accuracy,
)


def sample(
    qid: str,
    gold: str,
    skip_p_none: float,
    kb: dict[str, float] | None = None,
    answerable: bool = True,
    p_answered: float | None = None,
) -> CalibrationSample:
    return CalibrationSample(
        question_id=qid,
        gold_route=gold,
        answerable=answerable,
        skip_p_none=skip_p_none,
        kb=kb or {},
        p_answered=p_answered,
    )


class TestSkipThreshold:
    def test_separable_uses_gap_midpoint(self) -> None:
        samples = [
            sample("no1", "none", 0.80, {"policy": 0.5, "news": 0.5}, False),
            sample("no2", "none", 0.90, {"policy": 0.5, "news": 0.5}, False),
            sample("ans1", "policy", 0.20, {"policy": 0.8, "news": 0.2}),
            sample("ans2", "policy", 0.30, {"policy": 0.7, "news": 0.3}),
        ]
        thr = derive(samples)
        assert 0.30 < thr.skip < 0.80
        assert thr.skip == pytest.approx(0.55)

    def test_not_separable_prefers_no_retrieval(self) -> None:
        samples = [
            sample("no1", "none", 0.05, {"policy": 0.5, "news": 0.5}, False),
            sample("ans1", "policy", 0.95, {"policy": 0.9, "news": 0.1}),
        ]
        thr = derive(samples)
        assert thr.skip == pytest.approx(0.05)

    def test_no_questions_falls_back(self) -> None:
        assert derive([]).skip == FALLBACK_SKIP


class TestFanoutThreshold:
    def test_minimum_leader(self) -> None:
        samples = [
            sample("a", "policy", 0.1, {"policy": 0.60, "news": 0.40}),
            sample("b", "policy", 0.1, {"policy": 0.80, "news": 0.20}),
        ]
        thr = derive(samples)
        # kleinste Fuehrung 0.60 -> Schwelle knapp darunter
        assert thr.fanout == pytest.approx(0.599)

    def test_no_correct_leader_falls_back(self) -> None:
        samples = [sample("a", "policy", 0.1, {"policy": 0.1, "news": 0.9})]
        assert derive(samples).fanout == FALLBACK_FANOUT


class TestAnswerThreshold:
    def test_separable(self) -> None:
        samples = [
            sample("a", "policy", 0.1, {"policy": 1.0}, answerable=True, p_answered=0.9),
            sample("b", "policy", 0.1, {"policy": 1.0}, answerable=False, p_answered=0.1),
        ]
        assert derive(samples).answer == pytest.approx(0.5)

    def test_no_data_falls_back(self) -> None:
        assert derive([sample("a", "policy", 0.1, {"policy": 1.0})]).answer == FALLBACK_ANSWER


class TestCascade:
    def test_median_band(self) -> None:
        samples = [
            sample("a", "policy", 0.1, {"policy": 0.50, "news": 0.50}),
            sample("b", "policy", 0.1, {"policy": 0.60, "news": 0.40}),
            sample("c", "policy", 0.1, {"policy": 0.70, "news": 0.30}),
        ]
        thr = derive(samples)
        assert thr.cascade_lo == pytest.approx(0.50)
        assert thr.cascade_hi == pytest.approx(0.70)

    def test_empty_band_fallback(self) -> None:
        thr = derive([])
        assert (thr.cascade_lo, thr.cascade_hi) == pytest.approx((0.40, 0.60))


class TestRouteAccuracy:
    def test_perfect(self) -> None:
        samples = [
            sample("no", "none", 0.9, {"policy": 0.5, "news": 0.5}, False),
            sample("a", "policy", 0.1, {"policy": 0.8, "news": 0.2}),
        ]
        thr = derive(samples)
        assert route_accuracy(samples, thr) == pytest.approx(1.0)

    def test_picks_best_route(self) -> None:
        good = [
            sample("a", "policy", 0.1, {"policy": 0.8, "news": 0.2}),
            sample("no", "none", 0.9, {"policy": 0.5, "news": 0.5}, False),
        ]
        bad = [
            sample("a", "policy", 0.1, {"policy": 0.2, "news": 0.8}),
            sample("no", "none", 0.5, {"policy": 0.5, "news": 0.5}, False),
        ]
        samples = {"laya": bad, "slm": good}
        thr = {k: derive(v) for k, v in samples.items()}
        best = choose_best_route(samples, thr)
        assert best.name == "slm"

    def test_empty(self) -> None:
        assert route_accuracy([], derive([])) == 0.0


class TestProfile:
    def test_roundtrip(self) -> None:
        profile = CalibrationProfile(
            decision_route="slm",
            answer_backend="slm",
            skip=0.67,
            fanout=0.55,
            answer=0.5,
            cascade_lo=0.45,
            cascade_hi=0.65,
            created_at="2026-01-01T00:00:00Z",
            sources={"questions": 8},
        )
        restored = CalibrationProfile.from_dict(profile.to_dict())
        assert restored == profile

    def test_fallback_marked(self) -> None:
        profile = fallback_profile("slm", "slm")
        assert profile.fallback is True
        assert profile.skip == FALLBACK_SKIP
        assert profile.decision_route == "slm"

    def test_best_answer_backend(self) -> None:
        assert choose_best_answer_backend({"laya": 0.4, "slm": 0.9}).name == "slm"
        assert choose_best_answer_backend({}).name == "laya"
