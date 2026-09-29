"""Pipeline (P7): RagRouter = Decision + Retrieval + Answer-Check.

Einbindung in andere Systeme (Franks Kernvorgabe):
    config = load_config("config.yaml")
    router = RagRouter.from_config(config)   # echte Backends/Modelle
    result  = router.route_and_fetch("Wie lange dauert die Rueckgabe?")
    result.final     # answered | not_found | no_retrieval
    result.hits      # Evidenz-Chunks mit rag_key + Score-Detail
    result.decision  # Wahrscheinlichkeiten + Begründung
Oder komplett injiziert (Tests, eigene Backends):
    RagRouter.injected(config, decision=..., backends=..., checker=...)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rag_router.backends.base import RagBackend, SearchHit
from rag_router.backends.lancedb import registry
from rag_router.checking.base import AnswerChecker, CheckResult
from rag_router.config import RouterConfig
from rag_router.decision.base import DecisionBackend
from rag_router.router import DecisionThresholds, RouterDecisionEngine


@dataclass(frozen=True)
class RouterResult:
    """Ergebnis der Pipeline: Entscheidung, Treffer, Check, Endzustand."""

    decision: Any  # RouteDecision
    hits: list[SearchHit]
    check: CheckResult | None
    final: str  # "answered" | "not_found" | "no_retrieval"
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DecisionFactory(Protocol):
    def __call__(self, config: RouterConfig) -> DecisionBackend: ...


class RagRouter:
    """Die öffentliche Eintrittsklasse der Library."""

    def __init__(
        self,
        config: RouterConfig,
        decision: DecisionBackend,
        backends: Mapping[str, RagBackend],
        checker: AnswerChecker,
    ) -> None:
        self._config = config
        self._engine = RouterDecisionEngine(
            decision,
            thresholds=DecisionThresholds(
                skip=config.thresholds.skip,
                fanout=config.thresholds.fanout,
                answer=config.thresholds.answer,
            ),
        )
        self._engine.set_rag_descriptions(
            {key: rag.description for key, rag in config.rags.items()}
        )
        self._backends = dict(backends)
        self._checker = checker

    # -- Konstruktion ----------------------------------------------------

    @classmethod
    def injected(
        cls,
        config: RouterConfig,
        decision: DecisionBackend,
        backends: Mapping[str, RagBackend],
        checker: AnswerChecker,
    ) -> RagRouter:
        """Komplett injiziert (Tests, Hosts mit eigenen Backends).

        Backends muessen nicht fuer alle konfigurierten RAGs vorliegen —
        ein fehlendes Backend bricht erst beim Zugriff auf die geroutete
        Route mit einer klaren Meldung.
        """
        return cls(config, decision, dict(backends), checker)

    @classmethod
    def from_config(
        cls,
        config: RouterConfig,
        *,
        device: str | None = None,
    ) -> RagRouter:
        """Aus geladener Konfiguration: echte Backends + echte Modelle."""
        from rag_router.embedding import BgeM3Embedder

        embedder = BgeM3Embedder(
            model_name=config.defaults.embed.model,
            device=device or config.defaults.embed.device,
            batch_size=config.defaults.embed.batch_size,
        )
        backends: dict[str, RagBackend] = {}
        for key, rag in config.rags.items():
            backend_cls = registry[rag.backend.type]
            reranker = None
            if rag.rerank:
                reranker = _make_reranker()
            fts_language = getattr(config, "fts_language", "German")
            backends[key] = backend_cls(
                db_path=rag.backend.db,
                table=rag.backend.table,
                embedder=embedder,
                retriever=rag.retriever,
                rrf_k=config.thresholds.rrf_k,
                reranker=reranker,
                fts_language=fts_language,
            )
        decision = _make_decision(config)
        checker = _make_checker(config)
        return cls(config, decision, backends, checker)

    # -- Kern ------------------------------------------------------------

    def route(self, question: str) -> Any:
        """Nur die Routing-Entscheidung (ohne Suche)."""
        return self._engine.route(question)

    def route_and_fetch(self, question: str, top_k: int | None = None) -> RouterResult:
        """Routing -> (Fan-out-)Suche -> Answer-Check -> Ergebnis."""
        decision = self._engine.route(question)
        if decision.routes == ["none"]:
            return RouterResult(
                decision=decision, hits=[], check=None, final="no_retrieval"
            )
        all_hits: list[SearchHit] = []
        checks: list[CheckResult] = []
        for rag_key in decision.routes:
            rag_config = self._config.rags[rag_key]
            backend = self._backends[rag_key]
            hits = backend.search(question, top_k=top_k or rag_config.top_k)
            for hit in hits:
                all_hits.append(
                    SearchHit(
                        rag_key=rag_key,
                        doc_id=hit.doc_id,
                        text=hit.text,
                        score=hit.score,
                        scores=hit.scores,
                    )
                )
            if hits:
                passages = [hit.text for hit in hits]
                checks.append(self._checker.check(question, passages))
        if not checks:
            return RouterResult(
                decision=decision,
                hits=all_hits,
                check=None,
                final="not_found",
                detail={"reason": "keine Treffer"},
            )
        best = max(checks, key=lambda c: c.p_answered)
        final = (
            "answered"
            if best.p_answered >= self._config.thresholds.answer
            else "not_found"
        )
        return RouterResult(decision=decision, hits=all_hits, check=best, final=final)

    # -- Indexierung (Convenience) -------------------------------------------

    def index_texts(
        self, rag_key: str, texts: list[str], ids: list[str] | None = None
    ) -> int:
        """Dokumente in ein konfiguriertes RAG indexieren."""
        backend = self._backends[rag_key]
        return backend.index_texts(texts, ids=ids)

    def index_texts_from_file(self, rag_key: str, path: str | Path) -> int:
        """Dokumente aus JSON-Datei indexieren.

        Format: {"documents": [{"id": str, "text": str}, ...]} oder
        {"texts": [str, ...]}.
        """
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "documents" in data:
            documents = data["documents"]
            texts = [doc["text"] for doc in documents]
            ids = [str(doc.get("id", i)) for i, doc in enumerate(documents)]
        elif isinstance(data, dict) and "texts" in data:
            texts = [str(t) for t in data["texts"]]
            ids = None
        else:
            raise ValueError(
                f"{path}: erwartet {{'documents': [...]}} oder {{'texts': [...]}}"
            )
        return self.index_texts(rag_key, texts, ids=ids)


def _make_decision(config: RouterConfig) -> DecisionBackend:
    if config.backend == "laya":
        from rag_router.decision.laya import LayaDecisionBackend

        return LayaDecisionBackend.from_laya(
            model=config.laya.model if config.laya else "multilingual",
            max_len=config.laya.max_len if config.laya else 1024,
            preload=config.laya.preload if config.laya else False,
        )
    from rag_router.decision.llm import LlmDecisionBackend

    llm = config.llm
    if llm is None:
        raise ValueError("llm-Backend benoetigt router.llm-Konfiguration")
    return LlmDecisionBackend.from_settings(
        base_url=llm.base_url, api_key=llm.api_key, model=llm.model
    )


def _make_checker(config: RouterConfig) -> AnswerChecker:
    # Answer-Check folgt dem Decision-Backend (gleiche Runtime)
    if config.backend == "laya":
        from rag_router.checking.laya import LayaAnswerChecker

        return LayaAnswerChecker.from_laya(
            model=config.laya.model if config.laya else "multilingual",
            max_len=config.laya.max_len if config.laya else 1024,
            preload=config.laya.preload if config.laya else False,
        )
    from rag_router.checking.llm import LlmAnswerChecker

    llm = config.llm
    if llm is None:
        raise ValueError("llm-Checker benoetigt router.llm-Konfiguration")
    return LlmAnswerChecker.from_settings(
        base_url=llm.base_url, api_key=llm.api_key, model=llm.model
    )


def _make_reranker() -> Any:
    """bge-reranker-v2-m3 (FlagEmbedding), wie AurumVector."""
    from FlagEmbedding import FlagReranker

    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
