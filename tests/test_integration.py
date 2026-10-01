"""Integrationstests (P8/P10): echte Modelle, 32 originale deutsche Fragen.

Laeuft NUR mit `uv run pytest -m integration`. Erwartet: erstes Laden von
bge-m3 + laya (Download), danach CPU-Latenz im Sekundenbereich je Frage.

Drei Wege (vgl. docs/spec-decision-stages.md):
- TestLayaPipeline: reiner Laya-Weg, kein Endpunkt noetig. Belegt die
  dokumentierte Grenze S9: p(none) ist auf diesem deutschen Set invertiert,
  kein Schwellwert erfuellt beide Skip-Zusagen. Genau das begruendet slm/hybrid.
- TestAutoPipeline: Weg 'auto' (env-gated). Die Kalibrierung waehlt den besten
  Weg und die Schwellen; die vier Kern-Zusagen S7 werden geprueft.
- TestSlmRoute: der SLM-Weg kalibriert (env-gated).

SLM-Endpunkt/-Modell kommen aus der Umgebung (Fallback OLLAMA_*); ohne sie
skippen die endpoint-abhaengigen Tests mit Begruendung.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import pytest

from rag_router.config import load_config
from rag_router.pipeline import RagRouter

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE = Path(__file__).parent.parent / "config.example.yaml"

pytestmark = pytest.mark.integration


def _example_text(tmp_path: Path) -> str:
    text = EXAMPLE.read_text(encoding="utf-8")
    text = text.replace("db: ./data/lancedb", f"db: {tmp_path}/lancedb")
    # Embeddings optional ueber llama-swap (bge-m3-gguf) statt FlagEmbedding:
    base = os.environ.get("RR_ROUTER_EMBED_BASE_URL")
    if base:
        model = os.environ.get("RR_ROUTER_EMBED_MODEL", "bge-m3-gguf")
        text = text.replace(
            "      model: BAAI/bge-m3\n",
            f"      model: {model}\n      base_url: {base}\n",
            1,
        )
    return text


def _laya_config(tmp_path: Path) -> Path:
    target = tmp_path / "config_laya.yaml"
    target.write_text(_example_text(tmp_path), encoding="utf-8")
    return target


def _slm_env() -> tuple[str, str, str]:
    base = os.environ.get("RR_ROUTER_SLM_BASE_URL") or os.environ.get(
        "OLLAMA_BASE_URL", ""
    )
    key = os.environ.get("RR_ROUTER_SLM_API_KEY") or os.environ.get(
        "OLLAMA_API_KEY", "not-needed"
    )
    model = os.environ.get("RR_ROUTER_SLM_MODEL") or os.environ.get(
        "RR_ROUTER_LLM", ""
    )
    return base, key, model


def _slm_config(tmp_path: Path) -> Path:
    base, key, model = _slm_env()
    text = _example_text(tmp_path)
    text = text.replace("decision_route: laya", "decision_route: slm")
    slm_block = (
        "  slm:\n"
        f"    base_url: {base}\n"
        f"    api_key: \"{key or 'not-needed'}\"\n"
        f"    model: {model}\n"
    )
    text = text.replace("  laya:\n", slm_block + "  laya:\n", 1)
    target = tmp_path / "config_slm.yaml"
    target.write_text(text, encoding="utf-8")
    return target


def _auto_config(tmp_path: Path) -> Path:
    """Wie das Beispiel, aber decision_route/thresholds auf auto + SLM-Block."""
    base, key, model = _slm_env()
    text = _example_text(tmp_path)
    text = text.replace("decision_route: laya", "decision_route: auto")
    text = text.replace("skip: 0.60", "skip: auto")
    text = text.replace("fanout: 0.55", "fanout: auto")
    text = text.replace("answer: 0.50", "answer: auto")
    slm_block = (
        "  slm:\n"
        f"    base_url: {base}\n"
        f"    api_key: \"{key or 'not-needed'}\"\n"
        f"    model: {model}\n"
        "  calibration:\n"
        "    enabled: true\n"
        f"    cache: {tmp_path}/calibration.json\n"
    )
    text = text.replace("  laya:\n", slm_block + "  laya:\n", 1)
    target = tmp_path / "config_auto.yaml"
    target.write_text(text, encoding="utf-8")
    return target


def _index_corpus(router: RagRouter) -> None:
    corpus = json.loads((FIXTURES / "corpus_de.json").read_text(encoding="utf-8"))
    mapping = {"gem": "gemeinde", "fir": "firma", "die": "dienste"}
    buckets: dict[str, list[dict]] = {prefix: [] for prefix in mapping}
    for doc in corpus["documents"]:
        prefix = doc["id"][:3]
        if prefix not in buckets:
            raise AssertionError(f"unbekannter Korpus-Prefix: {doc['id']}")
        buckets[prefix].append(doc)
    for prefix, rag_key in mapping.items():
        docs = buckets[prefix]
        router.index_texts(
            rag_key, [d["text"] for d in docs], ids=[d["id"] for d in docs]
        )


QUESTIONS = json.loads((FIXTURES / "questions_de.json").read_text(encoding="utf-8"))[
    "questions"
]


def _questions(route: str | None, answerable: bool | None = None):
    out = []
    for q in QUESTIONS:
        if route is not None and q["gold_route"] != route:
            continue
        if answerable is not None and q["answerable"] != answerable:
            continue
        out.append(q)
    return out


@pytest.fixture(scope="module")
def laden(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("integ")
    config = load_config(_laya_config(tmp_path))
    router = RagRouter.from_config(config)
    _index_corpus(router)
    return router, tmp_path


class SlmGate:
    @staticmethod
    def skip_reason() -> str | None:
        base, _, model = _slm_env()
        if not base:
            return "RR_ROUTER_SLM_BASE_URL/OLLAMA_BASE_URL nicht gesetzt"
        if not model:
            return "RR_ROUTER_SLM_MODEL/RR_ROUTER_LLM nicht gesetzt"
        return None


class TestLayaPipeline:
    """Reiner Laya-Weg (S9): laeuft offline, belegt die Grenze."""

    def test_pipeline_runs(self, laden) -> None:
        router, _ = laden
        result = router.route_and_fetch("Wie lange ist die Leihfrist?")
        assert result.final in {"answered", "not_found", "no_retrieval"}

    def test_documented_laya_skip_inversion(self, laden) -> None:
        """S9: kein Schwellwert skippt alle no-retrieval UND keine echte Frage.

        p(none) der beantwortbaren Fragen reicht ueber das Minimum der
        no-retrieval-Fragen hinaus -> nicht separierbar. Genau diese Grenze
        begruendet die Wege slm/hybrid/auto.
        """
        router, _ = laden
        no_retrieval_min = min(
            router.route(q["question"]).distribution.get("none", 0.0)
            for q in _questions("none")
        )
        answerable_max = max(
            router.route(q["question"]).distribution.get("none", 0.0)
            for q in _questions(None, answerable=True)
        )
        assert no_retrieval_min < answerable_max, (
            "Set ist hier separierbar — die dokumentierte Laya-Grenze gilt fuer "
            "diese Fixture nicht mehr"
        )

    def test_median_latency_noted(self, laden) -> None:
        router, _ = laden
        times = []
        for q in _questions(None, answerable=True)[:9]:
            import time

            start = time.perf_counter()
            router.route_and_fetch(q["question"])
            times.append(time.perf_counter() - start)
        median = statistics.median(times)
        print(f"\nMEDIAN-Latenz Laya-Pipeline: {median:.2f}s")
        assert median < 30, f"Pipeline zu langsam: {median:.2f}s median"


@pytest.fixture(scope="module")
def auto_laden(tmp_path_factory):
    reason = SlmGate.skip_reason()
    if reason:
        pytest.skip(reason)
    tmp_path = tmp_path_factory.mktemp("auto_integ")
    config = load_config(_auto_config(tmp_path))
    router = RagRouter.from_config(config)
    _index_corpus(router)
    return router


class TestAutoPipeline:
    """Weg 'auto' (S7): kalibriert und erfuellt die vier Kern-Zusagen."""

    def test_calibration_profile_present(self, auto_laden) -> None:
        profile = auto_laden._calibration
        assert profile is not None, "auto muss kalibrieren"
        assert profile.decision_route in {"laya", "slm", "hybrid"}
        assert profile.fallback is False, "Endpoint erreichbar -> kein Fallback"
        for value in (profile.skip, profile.fanout, profile.answer):
            assert 0.0 <= value <= 1.0
        assert profile.sources, "Profil muss Quellen dokumentieren"

    def test_no_retrieval_all_skipped(self, auto_laden) -> None:
        router = auto_laden
        bad = [
            (q["id"], router.route(q["question"]).routes)
            for q in _questions("none")
            if router.route(q["question"]).routes != ["none"]
        ]
        assert not bad, f"kein Skip bei: {bad}"

    def test_no_false_skip_on_answerable(self, auto_laden) -> None:
        router = auto_laden
        skipped = [
            q["id"]
            for q in _questions(None, answerable=True)
            if "none" in router.route(q["question"]).routes
        ]
        assert not skipped, f"beantwortbare Frage faelschlich geskippt: {skipped}"

    def test_answerable_reach_gold_route(self, auto_laden) -> None:
        router = auto_laden
        hits = sum(
            1
            for q in _questions(None, answerable=True)
            if q["gold_route"] in router.route(q["question"]).routes
        )
        assert hits >= 12, f"nur {hits}/18 beantwortbare in der Gold-KB"

    def test_answerable_answered(self, auto_laden) -> None:
        router = auto_laden
        found = sum(
            1
            for q in _questions(None, answerable=True)
            if router.route_and_fetch(q["question"]).final == "answered"
        )
        assert found >= 12, f"nur {found}/18 beantwortbare als 'answered'"

    def test_unanswerable_not_answered(self, auto_laden) -> None:
        router = auto_laden
        wrong = [
            q["id"]
            for q in _questions(None, answerable=False)
            if router.route_and_fetch(q["question"]).final == "answered"
        ]
        assert len(wrong) <= 3, f"zu viele unbeantwortbare 'answered': {wrong}"


class TestSlmRoute:
    def test_slm_route_calibrates(self, tmp_path_factory) -> None:
        reason = SlmGate.skip_reason()
        if reason:
            pytest.skip(reason)
        from rag_router.pipeline import (
            _make_laya_route,
            _make_slm_route,
            _run_calibration,
        )

        tmp_path = tmp_path_factory.mktemp("cal_integ")
        config = load_config(_slm_config(tmp_path))
        rag_desc = {key: rag.description for key, rag in config.rags.items()}
        profile = _run_calibration(config, rag_desc)
        assert profile is not None, "Kalibrierung lieferte kein Profil"
        assert profile.answer_backend in {"laya", "slm"}
        assert profile.sources.get("best_route") in {"laya", "slm"}
        # Beide Wege sind baubar:
        assert _make_laya_route(config).name == "laya"
        assert _make_slm_route(config).name == "slm"
