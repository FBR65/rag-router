"""Konfigurationsmodell (P2): YAML -> gefrorene Dataclasses.

Alles, was der Router braucht, kommt aus einer YAML-Datei. Diese Datei
validiert konsequent und wirft ConfigError mit verstaendlicher Meldung,
sobald etwas fehlt oder ungueltig ist. Route "none" ist reserviert fuer
"No Retrieval" und kann nicht als RAG-Key verwendet werden.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_RETRIEVERS = ("dense", "sparse", "fts", "hybrid")
VALID_BACKENDS = ("laya", "llm")
VALID_DECISION_ROUTES = ("auto", "laya", "slm", "hybrid")
VALID_HYBRID_STRATEGIES = ("auto", "cascade", "committee")
VALID_AGGREGATES = ("mean", "product", "max", "primary")
VALID_LOGPROBS_MODES = ("auto", "logprobs", "text")
RAG_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RESERVED_RAG_KEYS = frozenset({"none"})
VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(RuntimeError):
    """Konfiguration unvollständig oder unbrauchbar."""


@dataclass(frozen=True)
class LayaConfig:
    model: str = "multilingual"
    max_len: int = 1024
    preload: bool = False


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class HybridConfig:
    strategy: str = "auto"
    aggregate: str = "mean"
    cascade_lo: float | None = None
    cascade_hi: float | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = True
    cache: str = "./.rag-router-calibration.json"
    questions: str | None = None
    max_questions: int = 64


@dataclass(frozen=True)
class Thresholds:
    skip: float | None = 0.60
    fanout: float | None = 0.55
    answer: float | None = 0.50
    rrf_k: int = 60



@dataclass(frozen=True)
class EmbedConfig:
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 16
    # Optional: llama-swap/OpenAI-kompatibler Endpunkt (dann laeuft bge-m3 im
    # llama-server statt im Prozess; kein torch-Modell, kein FlagEmbedding).
    base_url: str | None = None
    api_key: str = "not-needed"


@dataclass(frozen=True)
class Defaults:
    top_k: int = 5
    rerank: bool = True
    embed: EmbedConfig = EmbedConfig()


@dataclass(frozen=True)
class RagBackendConfig:
    type: str = "lancedb"
    db: str = ""
    table: str = ""


@dataclass(frozen=True)
class RagConfig:
    key: str
    description: str
    backend: RagBackendConfig
    retriever: str = "hybrid"
    top_k: int = 5
    rerank: bool = True


@dataclass(frozen=True)
class RouterConfig:
    backend: str
    laya: LayaConfig
    llm: LlmConfig | None
    thresholds: Thresholds
    defaults: Defaults
    rags: dict[str, RagConfig]
    decision_route: str = "auto"
    slm: LlmConfig | None = None
    hybrid: HybridConfig = HybridConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    answer_backend: str = "auto"

    def rag_keys(self) -> list[str]:
        """RAG-Keys in YAML-Reihenfolge (Laya-Optionen-Reihenfolge)."""
        return list(self.rags)


def _err(msg: str) -> ConfigError:
    return ConfigError(msg)


def _expand_vars(value: Any, where: str) -> Any:
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if resolved is None:
                raise _err(f"{where}: Umgebungsvariable {name} ist nicht gesetzt")
            return resolved

        return VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_vars(v, where) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_vars(v, where) for v in value]
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_map(data: Any, where: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise _err(f"{where}: erwartet ein Mapping, gefunden {type(data).__name__}")
    for key in data:
        if not isinstance(key, str):
            raise _err(f"{where}: Key ist kein String: {key!r}")
    return data


def _get_str(
    data: dict[str, Any], key: str, where: str, default: str | None = None
) -> str:
    if key not in data:
        if default is not None:
            return default
        raise _err(f"{where}: '{key}' fehlt")
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise _err(f"{where}: '{key}' muss nicht-leerer String sein")
    return value.strip()


def _get_bool(data: dict[str, Any], key: str, where: str, default: bool) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise _err(f"{where}: '{key}' muss bool sein")
    return value


def _get_int(data: dict[str, Any], key: str, where: str, default: int) -> int:
    if key not in data:
        return default
    value = data[key]
    if not _is_int(value):
        raise _err(f"{where}: '{key}' muss eine ganze Zahl sein")
    return value


def _get_float_01(data: dict[str, Any], key: str, where: str, default: float) -> float:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _err(f"{where}: '{key}' muss Zahl sein")
    if not 0.0 <= float(value) <= 1.0:
        raise _err(f"{where}: '{key}' muss in [0, 1] liegen, ist {value}")
    return float(value)


def _get_float_01_or_auto(
    data: dict[str, Any], key: str, where: str
) -> float | None:
    """Zahl in [0,1] oder 'auto' (-> None, wird kalibriert)."""
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _err(f"{where}: '{key}' muss Zahl oder 'auto' sein")
    if not 0.0 <= float(value) <= 1.0:
        raise _err(f"{where}: '{key}' muss in [0, 1] liegen, ist {value}")
    return float(value)


def _parse_laya(data: Any) -> LayaConfig:
    where = "router.laya"
    data = _require_map(data, where) if data is not None else {}
    data = data or {}
    known = {"model", "max_len", "preload"}
    unknown = set(data) - known
    if unknown:
        raise _err(f"{where}: unbekannte Felder {sorted(unknown)}")
    model = _get_str(data, "model", where, "multilingual")
    # Layapaket-Gruppennamen (laya.DEFAULT_MODELS): english|multilingual|
    # typed-decisions -> convaiinnovations/laya + Revision. Keine HF-ID.
    if model not in ("english", "multilingual", "typed-decisions"):
        raise _err(
            f"{where}.model: {model!r} ungueltig"
            " (verfuegbar: english, multilingual, typed-decisions)"
        )
    max_len = _get_int(data, "max_len", where, LayaConfig.max_len)
    if max_len not in (512, 1024):
        raise _err(f"{where}.max_len muss 512 oder 1024 sein, ist {max_len}")
    preload = _get_bool(data, "preload", where, False)
    return LayaConfig(model=model, max_len=max_len, preload=preload)


def _parse_llm(data: Any, where: str = "router.llm") -> LlmConfig:
    if data is None:
        raise _err(f"{where}: fehlt")
    data = _require_map(data, where)
    known = {"base_url", "api_key", "model"}
    unknown = set(data) - known
    if unknown:
        raise _err(f"{where}: unbekannte Felder {sorted(unknown)}")
    for field in ("base_url", "api_key", "model"):
        if field not in data:
            raise _err(f"{where}: '{field}' fehlt")
    return LlmConfig(
        base_url=_get_str(data, "base_url", where),
        api_key=_get_str(data, "api_key", where),
        model=_get_str(data, "model", where),
    )


def _parse_hybrid(data: Any) -> HybridConfig:
    where = "router.hybrid"
    data = _require_map(data, where) if data is not None else {}
    data = data or {}
    known = {"strategy", "aggregate", "cascade_lo", "cascade_hi"}
    unknown = set(data) - known
    if unknown:
        raise _err(f"{where}: unbekannte Felder {sorted(unknown)}")
    strategy = _get_str(data, "strategy", where, "auto")
    if strategy not in VALID_HYBRID_STRATEGIES:
        raise _err(
            f"{where}.strategy: {strategy!r} ungueltig"
            f" (verfuegbar: {', '.join(VALID_HYBRID_STRATEGIES)})"
        )
    aggregate = _get_str(data, "aggregate", where, "mean")
    if aggregate not in VALID_AGGREGATES:
        raise _err(
            f"{where}.aggregate: {aggregate!r} ungueltig"
            f" (verfuegbar: {', '.join(VALID_AGGREGATES)})"
        )
    lo = _get_float_01(data, "cascade_lo", where, 0.40)
    hi = _get_float_01(data, "cascade_hi", where, 0.60)
    if lo > hi:
        raise _err(f"{where}: cascade_lo ({lo}) > cascade_hi ({hi})")
    return HybridConfig(
        strategy=strategy, aggregate=aggregate, cascade_lo=lo, cascade_hi=hi
    )


def _parse_calibration(data: Any) -> CalibrationConfig:
    where = "router.calibration"
    data = _require_map(data, where) if data is not None else {}
    data = data or {}
    known = {"enabled", "cache", "questions", "max_questions"}
    unknown = set(data) - known
    if unknown:
        raise _err(f"{where}: unbekannte Felder {sorted(unknown)}")
    max_questions = _get_int(data, "max_questions", where, 64)
    if max_questions < 1:
        raise _err(f"{where}.max_questions muss >= 1 sein, ist {max_questions}")
    questions = data.get("questions")
    if questions is not None and (
        not isinstance(questions, str) or not questions.strip()
    ):
        raise _err(f"{where}.questions muss ein Pfad-String sein")
    return CalibrationConfig(
        enabled=_get_bool(data, "enabled", where, True),
        cache=_get_str(data, "cache", where, CalibrationConfig.cache),
        questions=questions.strip() if isinstance(questions, str) else None,
        max_questions=max_questions,
    )


def _parse_thresholds(data: Any) -> Thresholds:
    where = "router.thresholds"
    data = _require_map(data, where) if data is not None else {}
    data = data or {}
    known = {"skip", "fanout", "answer", "rrf_k"}
    unknown = set(data) - known
    if unknown:
        raise _err(f"{where}: unbekannte Felder {sorted(unknown)}")
    rrf_k = _get_int(data, "rrf_k", where, Thresholds.rrf_k)
    if rrf_k < 1:
        raise _err(f"{where}.rrf_k muss >= 1 sein, ist {rrf_k}")

    def _thr(key: str, default: float) -> float | None:
        if key not in data:
            return default
        return _get_float_01_or_auto(data, key, where)

    return Thresholds(
        skip=_thr("skip", 0.60),
        fanout=_thr("fanout", 0.55),
        answer=_thr("answer", 0.50),
        rrf_k=rrf_k,
    )


def _parse_embed(data: Any) -> EmbedConfig:
    where = "router.defaults.embed"
    data = _require_map(data, where) if data is not None else {}
    data = data or {}
    known = {"model", "device", "batch_size", "base_url", "api_key"}
    unknown = set(data) - known
    if unknown:
        raise _err(f"{where}: unbekannte Felder {sorted(unknown)}")
    batch_size = _get_int(data, "batch_size", where, EmbedConfig.batch_size)
    if batch_size < 1:
        raise _err(f"{where}.batch_size muss >= 1 sein, ist {batch_size}")
    base_url = (
        _get_str(data, "base_url", where, "")
        if data.get("base_url")
        else None
    )
    if base_url is not None and not base_url.startswith(("http://", "https://")):
        raise _err(f"{where}.base_url muss mit http:// oder https:// beginnen")
    return EmbedConfig(
        model=_get_str(data, "model", where, EmbedConfig.model),
        device=_get_str(data, "device", where, EmbedConfig.device),
        batch_size=batch_size,
        base_url=base_url,
        api_key=_get_str(data, "api_key", where, EmbedConfig.api_key),
    )


def _parse_defaults(data: Any) -> Defaults:
    where = "router.defaults"
    data = _require_map(data, where) if data is not None else {}
    data = data or {}
    known = {"top_k", "rerank", "embed"}
    unknown = set(data) - known
    if unknown:
        raise _err(f"{where}: unbekannte Felder {sorted(unknown)}")
    top_k = _get_int(data, "top_k", where, Defaults.top_k)
    if top_k < 1:
        raise _err(f"{where}.top_k muss >= 1 sein, ist {top_k}")
    return Defaults(
        top_k=top_k,
        rerank=_get_bool(data, "rerank", where, True),
        embed=_parse_embed(data.get("embed")),
    )


def _parse_rag(key: str, data: Any, defaults: Defaults) -> RagConfig:
    if not isinstance(data, dict):
        raise _err(f"rags.{key}: erwartet Mapping")
    known = {"description", "backend", "retriever", "top_k", "rerank"}
    unknown = set(data) - known
    if unknown:
        raise _err(f"rags.{key}: unbekannte Felder {sorted(unknown)}")
    description = _get_str(data, "description", f"rags.{key}")
    backend_data = data.get("backend")
    if not isinstance(backend_data, dict):
        raise _err(f"rags.{key}.backend: fehlt oder ist kein Mapping")
    backend_type = backend_data.get("type", "lancedb")
    known_backend = {"type", "db", "table"}
    unknown_backend = set(backend_data) - known_backend
    if unknown_backend:
        raise _err(f"rags.{key}.backend: unbekannte Felder {sorted(unknown_backend)}")
    if backend_type != "lancedb":
        raise _err(
            f"rags.{key}.backend.type: unbekannt {backend_type!r} (verfuegbar: lancedb)"
        )
    backend = RagBackendConfig(
        type="lancedb",
        db=_get_str(backend_data, "db", f"rags.{key}.backend"),
        table=_get_str(backend_data, "table", f"rags.{key}.backend"),
    )
    retriever = _get_str(data, "retriever", f"rags.{key}", "hybrid")
    if retriever not in VALID_RETRIEVERS:
        raise _err(
            f"rags.{key}.retriever: {retriever!r} ungueltig"
            f" (verfuegbar: {', '.join(VALID_RETRIEVERS)})"
        )
    top_k = _get_int(data, "top_k", f"rags.{key}", defaults.top_k)
    if top_k < 1:
        raise _err(f"rags.{key}.top_k muss >= 1 sein, ist {top_k}")
    return RagConfig(
        key=key,
        description=description,
        backend=backend,
        retriever=retriever,
        top_k=top_k,
        rerank=_get_bool(data, "rerank", f"rags.{key}", defaults.rerank),
    )


def parse_config(data: Any) -> RouterConfig:
    """Dict (expandiert) -> RouterConfig. Wirft ConfigError bei jedem Defekt."""
    root = _require_map(data, "Wurzel")
    unknown_root = set(root) - {"router", "rags"}
    if unknown_root:
        raise _err(
            f"unbekannte Sektionen {sorted(unknown_root)} (erwartet: rags, router)"
        )
    router_data = _require_map(root.get("router"), "router")
    known_router = {
        "decision_backend",
        "decision_route",
        "answer_backend",
        "language",
        "laya",
        "llm",
        "slm",
        "hybrid",
        "calibration",
        "thresholds",
        "defaults",
    }
    unknown_router = set(router_data) - known_router
    if unknown_router:
        raise _err(
            f"router: unbekannte Felder {sorted(unknown_router)}"
            " (erwartet: answer_backend, calibration, decision_backend,"
            " decision_route, defaults, hybrid, language, laya, llm, slm,"
            " thresholds)"
        )
    has_route = "decision_route" in router_data
    has_backend = "decision_backend" in router_data
    if has_route and has_backend:
        raise _err(
            "router.decision_route und router.decision_backend sind beide"
            " gesetzt — nur eines verwenden (decision_route ersetzt"
            " decision_backend)"
        )
    if not has_route and not has_backend:
        raise _err("router: 'decision_backend' oder 'decision_route' fehlt")
    if has_route:
        decision_route = _get_str(router_data, "decision_route", "router")
        if decision_route not in VALID_DECISION_ROUTES:
            raise _err(
                f"router.decision_route: {decision_route!r} ungueltig"
                f" (verfuegbar: {', '.join(VALID_DECISION_ROUTES)})"
            )
        backend = "laya" if decision_route in ("laya", "auto") else "llm"
    else:
        backend = _get_str(router_data, "decision_backend", "router")
        if backend not in VALID_BACKENDS:
            raise _err(
                f"router.decision_backend: {backend!r} ungueltig"
                f" (verfuegbar: {', '.join(VALID_BACKENDS)})"
            )
        decision_route = "laya" if backend == "laya" else "slm"
    answer_backend = _get_str(router_data, "answer_backend", "router", "auto")
    if answer_backend not in ("auto", "laya", "slm"):
        raise _err(
            f"router.answer_backend: {answer_backend!r} ungueltig"
            " (verfuegbar: auto, laya, slm)"
        )
    language = _get_str(router_data, "language", "router", "multilingual")
    if language not in ("multilingual", "english"):
        raise _err(
            f"router.language: {language!r} ungueltig"
            " (verfuegbar: multilingual, english)"
        )
    laya = _parse_laya(router_data.get("laya"))
    llm_raw = router_data.get("llm")
    slm_raw = router_data.get("slm")
    if slm_raw is not None and llm_raw is not None:
        raise _err("router.slm und router.llm sind beide gesetzt — nur eines")
    # slm ist der neue Name; llm bleibt Alias (gleiche Mechanik).
    slm = _parse_llm(slm_raw, "router.slm") if slm_raw is not None else None
    llm = _parse_llm(llm_raw, "router.llm") if llm_raw is not None else None
    if slm is None and llm is not None:
        slm = llm
    if has_backend and backend == "laya" and (llm_raw is not None or slm_raw is not None):
        raise _err(
            "router.slm/llm ist gesetzt, aber decision_backend=laya"
            " — Section entfernen oder backend auf llm setzen"
        )
    if decision_route in ("auto", "slm", "hybrid") and slm is None:
        raise _err(
            f"router.{'slm' if has_route else 'llm'}: fehlt, aber"
            f" decision_route={decision_route} braucht ein SLM"
        )
    hybrid = _parse_hybrid(router_data.get("hybrid"))
    calibration = _parse_calibration(router_data.get("calibration"))
    thresholds = _parse_thresholds(router_data.get("thresholds"))
    defaults = _parse_defaults(router_data.get("defaults"))
    rags_data = root.get("rags")
    if not isinstance(rags_data, dict) or not rags_data:
        raise _err("rags: fehlt oder ist leer (mindestens ein RAG nötig)")
    rags: dict[str, RagConfig] = {}
    for key, rag_data in rags_data.items():
        if not isinstance(key, str) or not RAG_KEY_PATTERN.match(key):
            raise _err(
                f"rags: Key {key!r} ungueltig (muss a-z, 0-9, Unterstrich, Start klein)"
            )
        if key in RESERVED_RAG_KEYS:
            raise _err(f"rags: Key {key!r} ist reserviert (No-Retrieval-Route)")
        if key in rags:
            raise _err(f"rags: Key {key!r} doppelt")
        rags[key] = _parse_rag(key, rag_data, defaults)
    return RouterConfig(
        backend=backend,
        laya=laya,
        llm=llm,
        thresholds=thresholds,
        defaults=defaults,
        rags=rags,
        decision_route=decision_route,
        slm=slm,
        hybrid=hybrid,
        calibration=calibration,
        answer_backend=answer_backend,
    )


def load_config(path: Path) -> RouterConfig:
    """YAML-Datei -> RouterConfig. Pfade relativ zur Datei aufloesen."""
    path = Path(path)
    if not path.is_file():
        raise _err(f"Konfigurationsdatei nicht gefunden: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise _err(f"Konfigurationsdatei unlesbar: {path}: {error}") from error
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise _err(f"Ungültiges YAML in {path}: {error}") from error
    data = _expand_vars(data, "konfig")
    config = parse_config(data)
    _resolve_rel_paths(config, path.parent)
    return config


def _resolve_rel_paths(config: RouterConfig, base: Path) -> None:
    """DB-Pfade relativ zur YAML-Datei interpretieren (mutabel nicht möglich:
    Dataclasses gefroren -> neu bauen)."""
    resolved: dict[str, RagConfig] = {}
    for key, rag in config.rags.items():
        db = Path(rag.backend.db)
        if not db.is_absolute():
            db = base / db
        resolved[key] = RagConfig(
            key=rag.key,
            description=rag.description,
            backend=RagBackendConfig(
                type=rag.backend.type,
                db=str(db),
                table=rag.backend.table,
            ),
            retriever=rag.retriever,
            top_k=rag.top_k,
            rerank=rag.rerank,
        )
    config.rags.clear()
    config.rags.update(resolved)
