# rag-router

Generischer RAG-Router (Python-Library): verteilt eine Frage an eines von N
frei per YAML konfigurierten RAGs — oder an „No Retrieval" (gar nicht suchen).
Routing-Logik nach „A RAG Router Built on Laya" im Design des Artikels
(vishalmysore/layaAsRagJudge): zwei getrennte Wahrscheinlichkeits-Entscheidungen
vor der Suche, ein Answer-Check danach:

1. **Skip:** p(none) ≥ 0.60 → es wird nicht gesucht (Plaudern/Trivialfragen).
2. **KB-Wahl:** renormalisiert über die RAGs; klare Führung ≥ 0.55 → allein
   durchsuchen, sonst Fan-out auf Top-2 (Gleichstände stabil in
   YAML-Reihenfolge). Grenzverhalten (FL-ULP) dokumentiert in
   `src/rag_router/router.py`.
3. **Answer-Check:** p(answered) ≥ 0.50 → „beantwortet", sonst `not_found`.

## Umfang

| Baustein | Stand |
|---|---|
| Konfig (YAML → Validierung, `${VAR}`-Expansion) | implementiert |
| Decision: Laya (`convaiinnovations/laya`, Gruppennamen `multilingual`/`english`/`typed-decisions`) | implementiert |
| Decision: OpenAI-kompatibles LLM (max_tokens=1, Logprob-A/B) | implementiert |
| Wissensbasen: LanceDB pro RAG, Retrieval `dense` \| `fts` \| `hybrid` (RRF), Upsert per `merge_insert` | implementiert |
| bge-m3 Embedding (dense, 1024 d), optionaler Reranker `bge-reranker-v2-m3` per `rerank:` | implementiert |
| Answer-Check: Laya (`noul`) oder LLM (Logprob yes/no) | implementiert |
| Pipeline `RagRouter.route_and_fetch()` + CLI `route`/`ask`/`index` | implementiert |
| Integrationstest: 32 originale deutsche Fragen, Laya + LLM-Backend | implementiert (Marker `integration`, nicht im Standard-Run) |
| bge-m3 **sparse** (lexical_weights) als eigener Retrieval-Modus | vorbereitet, nicht implementiert — FTS übernimmt die Keyword-Rolle |
| GAUNTLET (Mutationstests je Kernmodul, P9) | offen |

## Benutzung als Library

```python
from rag_router.config import load_config
from rag_router.pipeline import RagRouter

router = RagRouter.from_config(load_config("config.yaml"))
result = router.route_and_fetch("Wie lange dauert die Rueckgabe?")
result.final   # "answered" | "not_found" | "no_retrieval"
result.hits    # Evidenz-Chunks (rag_key, doc_id, text, score)
result.check.p_answered  # Wahrscheinlichkeit des Answer-Checks
```

Eigene Backends: `RagRouter.injected(config, decision=…, backends=…, checker=…)`
(vor allem für Tests).

## CLI (manuelles Testen)

```bash
uv sync
uv run rag-router index --config config.yaml --rag gemeinde --texts-file gemeinde.json
uv run rag-router route --config config.yaml "Wie viele Unterschriften braucht eine Petition?"
uv run rag-router ask   --config config.yaml --top-k 3 "Wie viel kostet das Mittagsgericht?"
```

Exit-Codes: 0 ok, 1 Fehler, 2 Konfiguration fehlt/ungültig. Vorbild-Konfig:
`config.example.yaml` (RAGs `gemeinde`/`firma`/`dienste`, Thresholds oben).

## Tests

```bash
uv run pytest                        # 74 Unit-Tests (keine Modelle)
uv run pytest -m integration -s      # + Integration: lädt bge-m3, laya, Reranker
```

Der Integration-LLM-Test braucht `OLLAMA_BASE_URL` (+ `OLLAMA_API_KEY`);
Modell per `RR_ROUTER_LLM` überschreibbar (Default `deepseek-v4.1-flash:cloud`).

## Status

P1–P8 implementiert und committed (74 Unit-Tests grün, ruff clean).
Offen: P9 GAUNTLET.