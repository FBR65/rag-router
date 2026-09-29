"""Integrationstests (P8): echte Modelle, 32 originale deutsche Fragen.

Laeuft NUR mit `uv run pytest -m integration` (pytest.ini schliesst sie im
Normal-Run aus). Erwartet: erstes Laden von bge-m3 + laya-multilingual
(Download), danach CPU-Latenz im Sekundenbereich je Frage.
Der LLM-Backend-Test benoetigt OLLAMA_BASE_URL + OLLAMA_API_KEY (Model
RR_ROUTER_LLM, Default: deepseek-v4.1-flash:cloud); ohne Env skippt er mit
Begruendung.
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

pytestmark = pytest.mark.integration


def _config_path(tmp_path):
    """config.example.yaml ins tmp kopieren (db-Pfade relativ umschrieben)."""
    example = Path(__file__).parent.parent / "config.example.yaml"
    target = tmp_path / "config.yaml"
    text = example.read_text(encoding="utf-8")
    # db in tmp innerhalb (keine Repo-Schmutz-Daten)
    text = text.replace("db: ./data/lancedb", f"db: {tmp_path}/lancedb")
    target.write_text(text, encoding="utf-8")
    return target


@pytest.fixture(scope="module")
def laden(tmp_path_factory):
    """Einmal pro Modul: bge-m3 + laya laden und alle 3 RAGs indizieren."""
    tmp_path = tmp_path_factory.mktemp("integ")
    cfg_path = _config_path(tmp_path)
    config = load_config(cfg_path)
    router = RagRouter.from_config(config)

    corpus = json.loads((FIXTURES / "corpus_de.json").read_text(encoding="utf-8"))
    # Keys und Lookup aus EINER Quelle: id-Prefix -> RAG-Key.
    mapping = {"gem": "gemeinde", "fir": "firma", "die": "dienste"}
    buckets: dict[str, list[dict]] = {prefix: [] for prefix in mapping}
    for doc in corpus["documents"]:
        prefix = doc["id"][:3]
        if prefix not in buckets:
            raise AssertionError(f"unbekannter Korpus-Prefix: {doc['id']}")
        buckets[prefix].append(doc)
    for prefix, rag_key in mapping.items():
        docs = buckets[prefix]
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        router.index_texts(rag_key, texts, ids=ids)
    return router, tmp_path


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


class TestLayaPipeline:
    def test_no_retrieval_questions(self, laden) -> None:
        router, _ = laden
        # Nur die 8 no-retrieval-Fragen: alle muessen auf "none" routen.
        routes = [
            (q["id"], router.route(q["question"]).routes) for q in _questions("none")
        ]
        bad = [(qid, r) for qid, r in routes if r != ["none"]]
        assert not bad, f"kein Skip bei: {bad}"

    def test_answerable_questions_reach_gold_route(self, laden) -> None:
        router, _ = laden
        hits_per_route = []
        for q in _questions(None, answerable=True):
            decision = router.route(q["question"])
            assert "none" not in decision.routes, f"{q['id']}: skip bei echter Frage"
            if q["gold_route"] in decision.routes:
                hits_per_route.append(q["id"])
        assert len(hits_per_route) >= 12, (
            f"nur {len(hits_per_route)} von 18 beantwortbaren Fragen landen "
            "in der richtigen Wissensbasis (Laya multilingual)"
        )

    def test_answer_check_passes_answerable(self, laden) -> None:
        router, _ = laden
        found = 0
        for q in _questions(None, answerable=True):
            result = router.route_and_fetch(q["question"])
            if result.final == "answered":
                found += 1
        assert found >= 12, (
            f"nur {found} von 18 beantwortbarem Fragen erkannt als 'answered'"
        )

    def test_unanswerable_do_not_get_answer(self, laden) -> None:
        router, _ = laden
        wrong = []
        for q in _questions(None, answerable=False):
            result = router.route_and_fetch(q["question"])
            if result.final == "answered":
                wrong.append(q["id"])
        assert len(wrong) <= 3, (
            f"zu viele unbeantwortbare Fragen fälschlich 'answered': {wrong}"
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


class OllamaGate:
    @staticmethod
    def skip_reason() -> str | None:
        env = os.environ
        if not env.get("OLLAMA_BASE_URL") and not env.get("OLLAMA_API_KEY"):
            return "OLLAMA_BASE_URL/OLLAMA_API_KEY nicht gesetzt"
        return None


class TestLlmPipeline:
    def test_llm_pipeline_with_ollama_cloud(
        self, monkeypatch, tmp_path_factory
    ) -> None:
        reason = OllamaGate.skip_reason()
        if reason:
            pytest.skip(reason)
        base = os.environ.get("OLLAMA_BASE_URL", "")
        key = os.environ.get("OLLAMA_API_KEY", "")
        model = os.environ.get("RR_ROUTER_LLM", "deepseek-v4.1-flash:cloud")
        # Eigenes config mit backend llm
        tmp_path = tmp_path_factory.mktemp("llminteg")
        example = Path(__file__).parent.parent / "config.example.yaml"
        text = example.read_text(encoding="utf-8")
        text = text.replace("decision_backend: laya", "decision_backend: llm")
        text = text.replace("db: ./data/lancedb", f"db: {tmp_path / 'lancedb'}")
        # laya-Block durch llm-Block ERSETZEN (gleiches level, keine
        # Duplicate-Keys, thresholds bleiben erhalten):
        import re

        text = re.sub(
            r"(  laya:\n(?:    .*\n)+)",
            (
                "  llm:\n"
                f"    base_url: {base}\n"
                f'    api_key: "{key or "not-needed"}"\n'
                f"    model: {model}\n"
            ),
            text,
            count=1,
        )
        target = tmp_path / "c.yaml"
        target.write_text(text, encoding="utf-8")

        from rag_router.config import ConfigError

        try:
            config = load_config(target)
        except ConfigError as error:
            pytest.skip(f"llm-Config nicht gueltig: {error}")
        router = RagRouter.from_config(config)
        corpus = json.loads((FIXTURES / "corpus_de.json").read_text(encoding="utf-8"))
        mapping = {"gem": "gemeinde", "fir": "firma", "die": "dienste"}
        for prefix, rag_key in mapping.items():
            docs = [d for d in corpus["documents"] if d["id"].startswith(prefix)]
            router.index_texts(
                rag_key, [d["text"] for d in docs], ids=[d["id"] for d in docs]
            )
        correct = 0
        for q in _questions(None, answerable=True)[:9]:
            result = router.route_and_fetch(q["question"])
            if result.final == "answered":
                correct += 1
        assert correct >= 5, (
            f"nur {correct}/9 beantwortbart auf LLM-Backend korrekt durchgelassen"
        )
