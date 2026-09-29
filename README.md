# rag-router

Generischer RAG-Router (Python-Library): verteilt eine Frage an eines von N
frei per YAML konfigurierten Wissensbasen (RAGs) — oder an „No Retrieval"
(gar nicht suchen). Es gibt **drei Entscheidungswege**, die per
`decision_route` gewählt werden:

| Weg | Mechanik | Vorteil |
|---|---|---|
| `laya` | Laya-Checkpoint, zwei getrennte Fragen (skip/choice) | lokal, kein Endpunkt |
| `slm` | OpenAI-kompatibles SLM, Logprobs je Entscheidung | trennt auch, wo Laya irrt |
| `hybrid` | beide (`cascade` oder `committee`) | bestes Ergebnis beider |
| `auto` | Messung wählt Weg + Schwellen selbst | nichts einstellen nötig |

Jeder Weg beantwortet **zwei getrennte Entscheidungen** nativ:

1. **Skip (D1):** `p(none) >= skip` → es wird nicht gesucht.
2. **KB-Wahl (D2):** nur unter den RAGs (kein `none`); klare Führung
   `>= fanout` → allein durchsuchen, sonst Fan-out auf Top-2. Der
   Answer-Check entscheidet danach.

> Warum getrennt? Auf einem reinen deutschen Fragensatz ist `p(none)` des
> Laya-`multilingual`-Checkpoints **invertiert**: kein einzelner Schwellwert
> erfüllt „alle No-Retrieval skippen" und „keine echte Frage skippen"
> gleichzeitig. Genau darum sind die Wege wählbar (Details:
> `docs/spec-decision-stages.md`).

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

Eigene Backends: `RagRouter.injected(config, decision=…, backends=…, checker=…)`.
Der Endpunkt/das Modell wird vom Nutzer gesetzt (generische `${VAR}`-Namen);
`auto` misst einmalig und kalibriert Schwellen, Wege und Answer-Backend selbst.

## Konfiguration (Kurz)

```yaml
router:
  decision_route: auto            # laya | slm | hybrid | auto
  answer_backend: auto            # auto | laya | slm
  slm:
    base_url: "${RR_ROUTER_SLM_BASE_URL}"
    api_key: "${RR_ROUTER_SLM_API_KEY}"
    model: "${RR_ROUTER_SLM_MODEL}"
  hybrid:
    strategy: auto                # auto | cascade | committee
    aggregate: mean               # committee: mean | product | max | primary
  calibration:
    enabled: true
    cache: ./.rag-router-calibration.json
  thresholds:
    skip: auto                    # auto | Zahl
    fanout: auto
    answer: auto
```

`decision_backend: laya|llm` bleibt als Alias erhalten (`llm` == `slm`).
Vollständiges Beispiel: `config.example.yaml`.

## CLI (manuelles Testen)

```bash
uv sync
uv run rag-router index --config config.yaml --rag gemeinde --texts-file gemeinde.json
uv run rag-router route --config config.yaml "Wie viele Unterschriften braucht eine Petition?"
uv run rag-router ask   --config config.yaml --top-k 3 "Wie viel kostet das Mittagsgericht?"
```

Exit-Codes: 0 ok, 1 Fehler, 2 Konfiguration fehlt/ungültig.

## Tests / Gauntlet

```bash
uv run pytest                        # 150 Unit-Tests (keine Modelle)
uv run pytest -m integration         # + echte Modelle (bge-m3, laya; SLM env-gated)
bash scripts/gauntlet.sh             # Gauntlet: Suite, lint, mypy, cov, mutation, integration
```

SLM-Tests brauchen `RR_ROUTER_SLM_BASE_URL` und `RR_ROUTER_SLM_MODEL`
(Fallback `OLLAMA_BASE_URL`/`RR_ROUTER_LLM`); ohne sie skippen sie mit
Begründung. Evidence-Report: `docs/evidence-decision-stages.md`.

## Status

Entscheidungswege (`laya`/`slm`/`hybrid`/`auto`), Kalibrierung (mit Datei-Lock
gegen Parallel-Läufe), Pipeline und CLI implementiert; 150 Unit-Tests +
10 Integrationstests grün, ruff clean, mypy clean, Mutation 9/9.
GAUNTLET-Report siehe `docs/evidence-decision-stages.md`.
