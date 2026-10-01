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

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rag_router.backends.base import RagBackend, SearchHit
from rag_router.backends.lancedb import registry
from rag_router.calibration import (
    FALLBACK_ANSWER,
    FALLBACK_FANOUT,
    FALLBACK_SKIP,
    CalibrationProfile,
    CalibrationSample,
    choose_best_route,
    derive,
)
from rag_router.checking.base import AnswerChecker, CheckResult
from rag_router.config import RouterConfig
from rag_router.decision.base import DecisionBackend
from rag_router.locking import ensure_cached_calibration
from rag_router.router import DecisionThresholds, RouterDecisionEngine

CALIBRATION_DATA = Path(__file__).parent / "data" / "calibration_de.json"


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
        decision: Any,
        backends: Mapping[str, RagBackend],
        checker: AnswerChecker,
        *,
        calibration: CalibrationProfile | None = None,
    ) -> None:
        self._config = config
        skip, fanout, answer = _resolve_thresholds(config, calibration)
        self._thresholds = DecisionThresholds(skip=skip, fanout=fanout, answer=answer)
        self._engine = RouterDecisionEngine(
            decision,
            thresholds=self._thresholds,
        )
        self._engine.set_rag_descriptions(
            {key: rag.description for key, rag in config.rags.items()}
        )
        self._backends = dict(backends)
        self._checker = checker
        self._calibration = calibration

    # -- Konstruktion ----------------------------------------------------

    @classmethod
    def injected(
        cls,
        config: RouterConfig,
        decision: Any,
        backends: Mapping[str, RagBackend],
        checker: AnswerChecker,
        *,
        calibration: CalibrationProfile | None = None,
    ) -> RagRouter:
        """Komplett injiziert (Tests, Hosts mit eigenen Backends).

        Backends muessen nicht fuer alle konfigurierten RAGs vorliegen —
        ein fehlendes Backend bricht erst beim Zugriff auf die geroutete
        Route mit einer klaren Meldung.
        """
        return cls(
            config, decision, dict(backends), checker, calibration=calibration
        )

    @classmethod
    def from_config(
        cls,
        config: RouterConfig,
        *,
        device: str | None = None,
    ) -> RagRouter:
        """Aus geladener Konfiguration: echte Backends + echte Modelle."""
        from rag_router.embedding import BgeM3Embedder, HttpEmbedder

        embed_cfg = config.defaults.embed
        if embed_cfg.base_url:
            embedder: Any = HttpEmbedder.from_settings(
                base_url=embed_cfg.base_url,
                model=embed_cfg.model,
                api_key=embed_cfg.api_key,
            )
        else:
            embedder = BgeM3Embedder(
                model_name=embed_cfg.model,
                device=device or embed_cfg.device,
                batch_size=embed_cfg.batch_size,
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
        decision, checker, calibration = _build_decision_and_checker(
            config, backends
        )
        return cls(
            config, decision, backends, checker, calibration=calibration
        )

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
            if best.p_answered >= self._thresholds.answer
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


def _make_laya_route(config: RouterConfig):
    from rag_router.decision.route import LayaRoute

    return LayaRoute.from_laya(
        model=config.laya.model if config.laya else "multilingual",
        max_len=config.laya.max_len if config.laya else 1024,
        preload=config.laya.preload if config.laya else False,
    )


def _make_slm_route(config: RouterConfig):
    from rag_router.decision.route import SlmRoute

    slm = config.slm or config.llm
    if slm is None:
        raise ValueError("slm-Weg benoetigt router.slm/llm-Konfiguration")
    return SlmRoute.from_settings(
        base_url=slm.base_url, api_key=slm.api_key, model=slm.model
    )


def _make_route(config: RouterConfig):
    """Baut den konfigurierten Weg (laya|slm|hybrid|auto->laya)."""
    from rag_router.decision.route import HybridRoute

    route_name = config.decision_route
    if route_name == "laya":
        return _make_laya_route(config)
    if route_name == "slm":
        return _make_slm_route(config)
    if route_name == "hybrid":
        return HybridRoute(
            _make_laya_route(config),
            _make_slm_route(config),
            strategy=config.hybrid.strategy if config.hybrid.strategy != "auto" else "cascade",
            aggregate=config.hybrid.aggregate,
            cascade_lo=config.hybrid.cascade_lo if config.hybrid.cascade_lo is not None else 0.40,
            cascade_hi=config.hybrid.cascade_hi if config.hybrid.cascade_hi is not None else 0.60,
        )
    # auto: Laya als Basis; die Kalibrierung darf auf slm/hybrid umschalten.
    return _make_laya_route(config)


def _make_checker_for(config: RouterConfig, backend: str) -> AnswerChecker:
    if backend == "laya":
        from rag_router.checking.laya import LayaAnswerChecker

        return LayaAnswerChecker.from_laya(
            model=config.laya.model if config.laya else "multilingual",
            max_len=config.laya.max_len if config.laya else 1024,
            preload=config.laya.preload if config.laya else False,
        )
    from rag_router.checking.llm import LlmAnswerChecker

    slm = config.slm or config.llm
    if slm is None:
        raise ValueError("slm-Checker benoetigt router.slm/llm-Konfiguration")
    return LlmAnswerChecker.from_settings(
        base_url=slm.base_url, api_key=slm.api_key, model=slm.model
    )


def _resolve_thresholds(
    config: RouterConfig, calibration: CalibrationProfile | None
) -> tuple[float, float, float]:
    """Zahlen aus Config; None ('auto') aus dem Profil oder Fallback."""
    skip = config.thresholds.skip
    fanout = config.thresholds.fanout
    answer = config.thresholds.answer
    if skip is None:
        skip = calibration.skip if calibration else FALLBACK_SKIP
    if fanout is None:
        fanout = calibration.fanout if calibration else FALLBACK_FANOUT
    if answer is None:
        answer = calibration.answer if calibration else FALLBACK_ANSWER
    return float(skip), float(fanout), float(answer)


def _load_calibration_questions(config: RouterConfig) -> list[dict[str, Any]]:
    path = config.calibration.questions
    source = Path(path) if path else CALIBRATION_DATA
    data = json.loads(source.read_text(encoding="utf-8"))
    questions = data["questions"]
    return questions[: config.calibration.max_questions]


def _measure_route(route, questions, rag_descriptions) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    for q in questions:
        needs = bool(q.get("needs_retrieval", False))
        gold = q.get("gold_route", "" if needs else "none")
        if needs and gold == "none":
            gold = ""  # Recherche erwartet, KB unbekannt
        skip = route.skip(q["question"], rag_descriptions)
        kb = route.choose(q["question"], rag_descriptions)
        samples.append(
            CalibrationSample(
                question_id=q["id"],
                gold_route=gold,
                answerable=bool(q.get("answerable", needs)),
                skip_p_none=float(skip.p_none),
                kb=dict(kb.probabilities),
            )
        )
    return samples


def _build_decision_and_checker(
    config: RouterConfig, backends: Mapping[str, RagBackend]
):
    """Baut Weg, Checker und (falls konfiguriert) das Kalibrierungsprofil.

    Faellt bei fehlender Messung auf den dokumentierten Kaltstart zurueck;
    der Router laeuft immer.
    """
    rag_descriptions = {key: rag.description for key, rag in config.rags.items()}
    cache = Path(config.calibration.cache)

    profile: CalibrationProfile | None = None
    if config.decision_route == "auto" and config.calibration.enabled:
        # Genau eine Messung bei kaltem Cache; wartende Prozesse lesen danach
        # das Ergebnis (Datei-Lock, S10). Nie blockierend im Fehlerfall.
        profile = ensure_cached_calibration(
            cache, lambda: _run_calibration(config, rag_descriptions)
        )

    try:
        route = _select_route(config, profile)
        route_name = getattr(route, "name", config.decision_route)
    except Exception:  # noqa: BLE001 — Kaltstart: Router darf nie blockieren
        route = _make_laya_route(config)
        route_name = "laya"
        profile = None

    if route_name == "legacy":
        route_name = "laya"
    answer_backend = _answer_backend(config, profile)
    # Auto-Schwellen aus dem Profil in die Config-Werte uebernehmen:
    checker = _make_checker_for(config, answer_backend)
    return route, checker, profile


def _select_route(config: RouterConfig, profile: CalibrationProfile | None):
    if config.decision_route != "auto" or profile is None:
        return _make_route(config)
    if profile.decision_route == "slm":
        return _make_slm_route(config)
    if profile.decision_route == "hybrid":
        from rag_router.decision.route import HybridRoute

        return HybridRoute(
            _make_laya_route(config),
            _make_slm_route(config),
            strategy="cascade",
            aggregate=config.hybrid.aggregate,
            cascade_lo=profile.cascade_lo,
            cascade_hi=profile.cascade_hi,
        )
    return _make_laya_route(config)


def _answer_backend(config: RouterConfig, profile: CalibrationProfile | None) -> str:
    if config.answer_backend != "auto":
        return config.answer_backend
    if profile is not None:
        return profile.answer_backend
    return "laya" if config.decision_route in ("laya", "auto") else "slm"


def _run_calibration(
    config: RouterConfig, rag_descriptions: Mapping[str, str]
) -> CalibrationProfile | None:
    """Einmalige Messung aller verfuegbaren Wege -> Profil (oder None)."""
    try:
        questions = _load_calibration_questions(config)
    except (OSError, ValueError, KeyError):
        return None
    routes = {"laya": _make_laya_route(config)}
    if config.slm is not None or config.llm is not None:
        routes["slm"] = _make_slm_route(config)
    samples: dict[str, list[CalibrationSample]] = {}
    thresholds: dict[str, Any] = {}
    try:
        for name, route in routes.items():
            route_samples = _measure_route(route, questions, rag_descriptions)
            samples[name] = route_samples
            thresholds[name] = derive(route_samples)
    except Exception:  # noqa: BLE001 — Messung darf den Start nicht brechen
        return None
    best = choose_best_route(samples, thresholds)
    thr = thresholds[best.name]
    # answer_backend folgt dem besten Weg (einfach + deterministisch).
    answer_backend = "slm" if best.name == "slm" else "laya"
    from datetime import datetime

    return CalibrationProfile(
        decision_route=best.name,
        answer_backend=answer_backend,
        skip=thr.skip,
        fanout=thr.fanout,
        answer=thr.answer,
        cascade_lo=thr.cascade_lo,
        cascade_hi=thr.cascade_hi,
        created_at=datetime.now(UTC).isoformat(),
        sources={
            "best_route": best.name,
            "route_accuracy": best.accuracy,
            "candidates": sorted(routes),
            "derived": dict(thr.sources),
        },
    )


def _make_reranker() -> Any:
    """bge-reranker-v2-m3 (FlagEmbedding), wie AurumVector."""
    from FlagEmbedding import FlagReranker

    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
