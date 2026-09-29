"""Tests fuer das Konfigurationsmodell (P2)."""

import pytest

from rag_router.config import ConfigError, load_config

BASE_YAML = """
router:
  decision_backend: laya
  laya:
    model: multilingual
    max_len: 1024
rags:
  policy:
    description: "Richtlinien und Hausregeln"
    backend:
      type: lancedb
      db: ./data/lancedb
      table: policy_chunks
    retriever: hybrid
    top_k: 3
  news:
    description: "Nachrichten und Ereignisse"
    backend:
      db: ./data/lancedb
      table: news_chunks
"""


def write(tmp_path, text: str):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_minimal_config_loads_with_defaults(tmp_path) -> None:
    cfg = load_config(write(tmp_path, BASE_YAML))

    assert cfg.backend == "laya"
    assert cfg.laya.model == "multilingual"
    assert cfg.laya.max_len == 1024
    assert cfg.laya.preload is False
    assert cfg.thresholds.skip == 0.60
    assert cfg.thresholds.fanout == 0.55
    assert cfg.thresholds.answer == 0.50
    assert cfg.thresholds.rrf_k == 60
    assert cfg.defaults.top_k == 5
    assert cfg.defaults.rerank is True
    assert cfg.defaults.embed.model == "BAAI/bge-m3"
    assert cfg.defaults.embed.device == "cpu"

    # Reihenfolge = YAML-Reihenfolge (Reihenfolge ist Laya-Optionsreihenfolge)
    assert list(cfg.rags) == ["policy", "news"]
    assert cfg.rag_keys() == ["policy", "news"]
    assert cfg.rags["policy"].retriever == "hybrid"
    assert cfg.rags["news"].retriever == "hybrid"  # Default
    assert cfg.rags["news"].top_k == 5  # Default aus defaults.top_k
    assert cfg.rags["news"].rerank is True
    assert cfg.rags["policy"].backend.type == "lancedb"
    assert cfg.rags["policy"].backend.table == "policy_chunks"


def test_backend_llm_requires_llm_section(tmp_path) -> None:
    yml = BASE_YAML.replace("decision_backend: laya", "decision_backend: llm")
    with pytest.raises(ConfigError, match="llm"):
        load_config(write(tmp_path, yml))


def test_llm_backend_config_loads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RR_TEST_BASE", "https://api.example.org/v1")
    monkeypatch.setenv("RR_TEST_KEY", "k-123")
    yml = """
router:
  decision_backend: llm
  llm:
    base_url: ${RR_TEST_BASE}
    api_key: ${RR_TEST_KEY}
    model: test-model
rags:
  policy:
    description: "Richtlinien und Hausregeln"
    backend:
      db: ./data/lancedb
      table: policy_chunks
"""
    cfg = load_config(write(tmp_path, yml))
    assert cfg.backend == "llm"
    assert cfg.llm.base_url == "https://api.example.org/v1"
    assert cfg.llm.api_key == "k-123"
    assert cfg.llm.model == "test-model"


def test_missing_env_var_is_clear_error(tmp_path) -> None:
    yml = """
router:
  decision_backend: llm
  llm:
    base_url: ${RR_DEFINITIV_NICHT_GESETZT}
    api_key: k
    model: m
rags:
  policy:
    description: "x"
    backend:
      db: ./d
      table: t
"""
    with pytest.raises(ConfigError, match="RR_DEFINITIV_NICHT_GESETZT"):
        load_config(write(tmp_path, yml))


def test_invalid_backend_rejected(tmp_path) -> None:
    yml = BASE_YAML.replace("decision_backend: laya", "decision_backend: jev")
    with pytest.raises(ConfigError, match="decision_backend"):
        load_config(write(tmp_path, yml))


def test_invalid_retriever_rejected(tmp_path) -> None:
    yml = BASE_YAML.replace("retriever: hybrid", "retriever: magisch")
    with pytest.raises(ConfigError, match="retriever"):
        load_config(write(tmp_path, yml))


def test_threshold_out_of_range_rejected(tmp_path) -> None:
    # Ohne thresholds-Sektion gilt der Default (0.60):
    cfg = load_config(write(tmp_path, BASE_YAML))
    assert cfg.thresholds.skip == 0.60

    # Ausserhalb [0, 1] -> klarer Fehler:
    yml2 = """
router:
  decision_backend: laya
  thresholds:
    skip: 1.5
rags:
  policy:
    description: "x"
    backend:
      db: ./d
      table: t
"""
    with pytest.raises(ConfigError, match="skip"):
        load_config(write(tmp_path, yml2))


def test_rag_named_none_is_reserved(tmp_path) -> None:
    yml = BASE_YAML.replace("  policy:", "  none:")
    with pytest.raises(ConfigError, match="none"):
        load_config(write(tmp_path, yml))


def test_empty_rags_rejected(tmp_path) -> None:
    yml = BASE_YAML.split("rags:")[0] + "rags: {}\n"
    with pytest.raises(ConfigError, match="rags"):
        load_config(write(tmp_path, yml))


def test_unknown_top_level_key_rejected(tmp_path) -> None:
    yml = BASE_YAML + "extra: wahr\n"
    with pytest.raises(ConfigError, match="extra"):
        load_config(write(tmp_path, yml))


def test_missing_description_rejected(tmp_path) -> None:
    yml = BASE_YAML.replace('    description: "Richtlinien und Hausregeln"\n', "")
    with pytest.raises(ConfigError, match="description"):
        load_config(write(tmp_path, yml))


def test_missing_file_is_clear_error(tmp_path) -> None:
    with pytest.raises(ConfigError, match="nicht gefunden"):
        load_config(tmp_path / "gibt_es_nicht.yaml")


def test_invalid_yaml_is_clear_error(tmp_path) -> None:
    p = write(tmp_path, "router: [unclosed")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(p)


def test_per_rag_overrides(tmp_path) -> None:
    yml = """
router:
  decision_backend: laya
rags:
  a:
    description: "Alpha"
    backend:
      db: ./d
      table: ta
    retriever: dense
    top_k: 7
    rerank: false
  b:
    description: "Beta"
    backend:
      db: ./d
      table: tb
"""
    cfg = load_config(write(tmp_path, yml))
    assert cfg.rags["a"].retriever == "dense"
    assert cfg.rags["a"].top_k == 7
    assert cfg.rags["a"].rerank is False
    assert cfg.rags["b"].rerank is True  # Default


SLM_BLOCK = """
  slm:
    base_url: ${RR_TEST_BASE}
    api_key: ${RR_TEST_KEY}
    model: test-model
"""


def _with_slm(yml: str) -> str:
    return yml.replace(
        "  laya:\n    model: multilingual\n    max_len: 1024\n",
        "  laya:\n    model: multilingual\n    max_len: 1024\n" + SLM_BLOCK,
    )


def test_decision_route_slm(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RR_TEST_BASE", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("RR_TEST_KEY", "k")
    yml = _with_slm(BASE_YAML).replace(
        "  decision_backend: laya", "  decision_route: slm"
    )
    cfg = load_config(write(tmp_path, yml))
    assert cfg.decision_route == "slm"
    assert cfg.slm.model == "test-model"
    assert cfg.slm.base_url == "http://127.0.0.1:8080/v1"


def test_decision_route_hybrid_needs_slm(tmp_path) -> None:
    yml = BASE_YAML.replace("  decision_backend: laya", "  decision_route: hybrid")
    with pytest.raises(ConfigError, match="slm"):
        load_config(write(tmp_path, yml))


def test_decision_route_invalid_rejected(tmp_path) -> None:
    yml = BASE_YAML.replace("  decision_backend: laya", "  decision_route: magisch")
    with pytest.raises(ConfigError, match="decision_route"):
        load_config(write(tmp_path, yml))


def test_route_and_backend_conflict_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RR_TEST_BASE", "http://x/v1")
    monkeypatch.setenv("RR_TEST_KEY", "k")
    yml = _with_slm(BASE_YAML).replace(
        "  decision_backend: laya", "  decision_backend: laya\n  decision_route: slm"
    )
    with pytest.raises(ConfigError, match="decision_route"):
        load_config(write(tmp_path, yml))


def test_thresholds_auto_is_none(tmp_path) -> None:
    yml = BASE_YAML + "  thresholds:\n    skip: auto\n    fanout: auto\n    answer: auto\n"
    # thresholds gehoert unter router -> korrekt einsetzen
    yml = BASE_YAML.replace(
        "  laya:\n    model: multilingual\n    max_len: 1024\n",
        "  thresholds:\n    skip: auto\n    fanout: 0.55\n    answer: auto\n",
    )
    cfg = load_config(write(tmp_path, yml))
    assert cfg.thresholds.skip is None
    assert cfg.thresholds.fanout == 0.55
    assert cfg.thresholds.answer is None


def test_thresholds_numbers_still_work(tmp_path) -> None:
    cfg = load_config(write(tmp_path, BASE_YAML))
    assert cfg.thresholds.skip == 0.60


def test_slm_key_alias_accepted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RR_TEST_BASE", "http://x/v1")
    monkeypatch.setenv("RR_TEST_KEY", "k")
    yml = _with_slm(BASE_YAML).replace(
        "  decision_backend: laya", "  decision_backend: llm"
    )
    cfg = load_config(write(tmp_path, yml))
    assert cfg.backend == "llm"
    assert cfg.slm.model == "test-model"


def test_hybrid_aggregate_invalid_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RR_TEST_BASE", "http://x/v1")
    monkeypatch.setenv("RR_TEST_KEY", "k")
    yml = _with_slm(BASE_YAML).replace(
        "  decision_backend: laya", "  decision_route: hybrid"
    )
    yml = yml.replace(
        "  laya:\n",
        "  hybrid:\n    strategy: committee\n    aggregate: magisch\n  laya:\n",
    )
    with pytest.raises(ConfigError, match="aggregate"):
        load_config(write(tmp_path, yml))


def test_calibration_section_loads(tmp_path) -> None:
    yml = BASE_YAML.replace(
        "  laya:\n",
        "  calibration:\n    enabled: false\n    cache: /tmp/prof.json\n  laya:\n",
    )
    cfg = load_config(write(tmp_path, yml))
    assert cfg.calibration.enabled is False
    assert cfg.calibration.cache == "/tmp/prof.json"


def test_unknown_slm_field_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RR_TEST_BASE", "http://x/v1")
    monkeypatch.setenv("RR_TEST_KEY", "k")
    yml = _with_slm(BASE_YAML).replace(
        "    model: test-model\n",
        "    model: test-model\n    magie: 1\n",
    )
    with pytest.raises(ConfigError, match="magie"):
        load_config(write(tmp_path, yml))
