"""Verifiziert die llm-Config-Transformation aus dem Integrationstest einmal
statisch (ohne Endpunkt und ohne Env)."""

import re
import tempfile
from pathlib import Path

from rag_router.config import load_config

text = (Path(__file__).parent.parent / "config.example.yaml").read_text(
    encoding="utf-8"
)
text = text.replace("decision_backend: laya", "decision_backend: llm")
out = re.sub(
    r"(  laya:\n(?:    .*\n)+)",
    '  llm:\n    base_url: http://x\n    api_key: "k"\n    model: m\n',
    text,
    count=1,
)
p = Path(tempfile.mkdtemp()) / "c.yaml"
p.write_text(
    out.replace("db: ./data/lancedb", "db: /tmp/rr_probe_lncdb"), encoding="utf-8"
)
cfg = load_config(p)
print("backend:", cfg.backend)
print("llm.model:", cfg.llm.model if cfg.llm else None)
print("rags:", len(cfg.rags), sorted(cfg.rags))
print("skip:", cfg.thresholds.skip, "fanout:", cfg.thresholds.fanout)
