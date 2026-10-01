# EVIDENCE: Entscheidungswege (laya | slm | hybrid | auto)

**Stand:** 2026-09-29 · **Source-State:** Commit `2a9f174` (Arbeitsbaum danach
unverändert) · **Spec:** `docs/spec-decision-stages.md` (Rev. **4**, freigegeben;
Rev. 4 = Nutzerauftrag „Behebe 2 und 4": Typenchecker + Kalibrierungs-Lock).
Spec-Freigabe der Umgebungsänderung (mypy) erteilt.

## Umfang (Tier)

**Tier 3 (hoch)** nach Kalibrierung: neue öffentliche Schnittstellen, Routing-
Entscheidung (Kern des Produkts), Automatik mit Persistenz + Nebenläufigkeit.
Failure-Model: falsches Skip einer echten Frage, stiller Fallback, unerreichbarer
Endpoint, `auto`-Werte außerhalb [0,1], Kalibrierungs-Overfit, **parallele
Kalibrierung zweier Prozesse**, Typfehler zur Laufzeit.

## Spec → Test-Mapping

| Kriterium | Test |
|---|---|
| S1 LayaRoute D1/D2 nativ | `test_decision_routes.py::TestLayaRoute` |
| S2 SlmRoute D1/D2 nativ | `test_decision_routes.py::TestSlmRoute` |
| S3 KbDist ohne none | `TestLayaRoute::test_choose_has_no_none`, `TestSlmRoute::test_choose_*` |
| S4 Hybrid cascade/committee | `TestHybridCascade`, `TestHybridCommittee` |
| S5 Rückwärtskompatibel | `test_router.py` (21), `test_pipeline.py::Test*` Legacy-Pfad |
| S6 Validierung | `test_config.py::test_decision_route_*`, `test_*_rejected` |
| S7 Integration (auto) | `test_integration.py::TestAutoPipeline` (6) |
| S8 Automatik | `test_calibration.py` (20), `test_pipeline.py::TestAutoCalibration` |
| S9 Laya-Grenze | `TestLayaPipeline::test_documented_laya_skip_inversion` |
| S10 Kalibrierungs-Lock | `test_locking.py` (11) |
| S11 Statische Typen | `uv run mypy` (Baseline aus Rev. 4) |

Alle Szenarien der Spec sind auf grüne Tests abgebildet; keine Lücke.

## Gauntlet-Layer (frischer Lauf nach letzter Code-Änderung, Commit 2a9f174)

| Layer | Kommando | Ergebnis |
|---|---|---|
| Volle Suite (Unit) | `uv run pytest -q` | **150 passed**, 10 deselected |
| Statische Typen | `uv run mypy` | **Success: no issues found in 20 source files** (Baseline 5 → 0) |
| Lint/Format | `uv run ruff check src/ tests/` | **All checks passed!** |
| Coverage (Unit) | `uv run pytest --cov=rag_router` | gesamt 83 %; neue Module u. a. `calibration` 97 %, `locking` hoch, `decision/route` 86 % |
| Integration (echtes Modell) | `RR_ROUTER_SLM_* uv run pytest -m integration` | **10 passed** |
| Mutation (manuell, 9 Mutanten) | `bash scripts/mutation_check.sh` | **9/9 getötet** |
| Property-Tests | `test_calibration.py::TestPropertiesSweep` | 2 Properties, 400 Zufalls-Eingaben; `hypothesis` nicht installiert, stdlib-Sweep als Ersatz |
| Complexity | Sichtprüfung | neue Funktionen klein/ein Zweck; `locking.py` < 140 Zeilen |
| Real Execution | CLI `index`+`route`+`ask` am Endpoint | `auto` skippt Mathe, sucht echte Frage |
| Supply chain | `pyproject.toml` | **neu: mypy** (nur dev, `uv sync`; kein Laufzeit-Impact); geprüft |
| Suite-Health | 3× randomisiert (`pytest-randomly`) | 3× grün, keine Flakes |

### Integration-Details (S7)

Modell: `Qwen2.5-Coder-7B-Instruct-heretic` @ `http://127.0.0.1:8080/v1`
(SLM-Endpoint). `TestAutoPipeline` nach Kalibrierung: kein-retrieval 8/8 → `none`,
kein falsches Skip, ≥12/18 Gold-KB, ≥12/18 „answered", ≤3 fälschlich beantwortet.

### Real Execution (CLI, echter Endpoint)

```
$ rag-router route ... "Was ist 17 mal 23?"        -> routes=[none] reason=skip (0.999)
$ rag-router route ... "Wann ist die Kernzeit...?" -> routes=[firma] clear_leader
$ rag-router ask   ... "Wie viel kostet das Mittagsgericht?" (auto)
      -> routes=[gemeinde], hits=3, p_answered=0.022, final=not_found
```
Kaltstart-Fall (feste `skip: 0.60`) skippte „Mittagsgericht" fälschlich;
`auto` kalibrierte `skip=0.881` und suchte korrekt. Belegt den Wert der Automatik.

## Gefundene und behobene Fehler (ehrlich)

1. **`auto`-Schwellen brachen `route_and_fetch`** (`float >= None`). Der
   Integrationstest deckte es auf; Fix: aufgelöste Schwellen in `self._thresholds`.
   Regressionstest `test_route_and_fetch_uses_calibrated_answer`.
2. **SLM-Skip-Prompt war abstrakt** („Dokumente?") und trennte auf dem deutschen
   Set nicht (Coder-Modell: 12/18 falsche Skips). Umbau auf einen Choice-Prompt
   über die **konkreten KB-Beschreibungen + none** → 0/18 falsche Skips.
3. **Skip-Schwelle in der Lückenmitte** war ausreißeranfällig (`cal-doc-9`=0.97).
   Fix: knapp über dem höchsten beantwortbaren Wert. Mutanten `skip_thr_mid`/`le`
   getötet.
4. **CLI-Entry-Point war ein Dummy** (Regel 8, vorbestehend): `rag_router.main`
   gab nur die Version aus. Fix + Regressionstests.

## Bekannte Grenzen (nicht behauptet)

- **Laya allein erfüllt S7 nicht** (bewusst, S9): `p(none)` ist auf dem
  deutschen Set invertiert; nur `slm`/`hybrid`/`auto` erfüllen die Kernzusagen.
- **`hypothesis` fehlt** weiterhin; Properties als stdlib-Sweep, schwächer als
  ein echtes PBT-Tool (kein automatisches Schrumpfen). Bewusst offen gelassen
  (kein Nutzerauftrag); `mypy` wurde auf Auftrag nachgerüstet.
- **`_run_calibration` ist teuer** (ein Forward-Pass je Frage und Weg). Das
  Lock verhindert zwar doppelte Parallel-Läufe, macht die Einzelmessung aber
  nicht schneller; sie läuft nur einmalig, danach aus dem Cache.
- Der Kalibrierungssatz ist **generisch** (nur `needs_retrieval`), daher
  kalibriert `fanout`/`answer` auf Fallback, wenn kein KB-Gold vorliegt.

**Rev. 4 – geschlossen:** Typenchecker (mypy, Baseline 5 → 0, Layer läuft) und
Nebenläufigkeits-Schutz (Datei-Lock, genau eine Messung, S10) sind umgesetzt.

## Reproduzierbarkeit

- Dev-Versionen: pytest 9.1.1, pytest-randomly 5.0.0, pytest-cov 7.1.0,
  ruff 0.16.9, mypy 2.3.1 (aus `pyproject.toml`).
- Mutationsskript persistiert: `scripts/mutation_check.sh` (9 Mutanten,
  stellt den Arbeitsbaum nach jedem Mutanten wieder her).
- Ein Einstiegskommando (siehe `scripts/gauntlet.sh`).
- Ohne Endpoint: `uv run pytest -m "not integration"` (158) und
  `uv run pytest -m integration` skippt die SLM-Tests mit Begründung.

---

## Nachtrag 2026-10-01: Lauf gegen llama-swap (alle Modelle auf `:8080`)

Alle Modelle laufen jetzt über den llama-swap-Container
(`http://127.0.0.1:8080/v1`): `bge-m3-gguf` (Embeddings), SLM, Laya lokal.

**Geänderte Umgebung (nicht Repo-Code):** `bge-m3-gguf` in
`/home/speedy/Dokumente/dev/container/llama_swap/config.yaml` startet den
llama-server nun mit `--embeddings`; ohne das antwortet `/v1/embeddings` mit
HTTP 501 (`This server does not support embeddings`). Podman-Container
`llama-swapel` wurde neu gestartet.

**Neuer Code:** `router.defaults.embed.base_url`/`api_key` (optional). Ist
`base_url` gesetzt, nutzt `RagRouter.from_config` `HttpEmbedder`
(`/v1/embeddings`) statt `BgeM3Embedder`/FlagEmbedding im Prozess. Das ist
dasselbe Modell (bge-m3), verifiziert: cos(Endpoint, lokal) = 0.9995 bei
identischem Ranking, beide 1024-dim und normiert. Tests: `TestHttpEmbedder`
(3) + `test_config` Embed-Endpunkt (5).

### Ergebnis (bge-m3-gguf über llama-swap)

| Lauf | Embeddings | SLM (`RR_ROUTER_SLM_MODEL`) | Ergebnis |
|---|---|---|---|
| A | lokal (FlagEmbedding) | Qwen2.5-Coder-7B | 10 passed |
| B | lokal | Qwen2.5-7B-Instruct | **1 failed**: `test_no_false_skip_on_answerable` |
| C | lokal | gemma-4-12B-TurboQuant | 10 passed |
| D | **llama-swap** | Qwen2.5-7B-Instruct | **1 failed**: dito |
| E | **llama-swap** | gemma-4-12B-TurboQuant | 10 passed (567 s) |
| F | **llama-swap** | *(kein SLM)* | `TestLayaPipeline` 3 passed, 7 skipped; Median-Latenz 0.69 s (vorher 0.95 s mit lokalem bge-m3) |

**Belegte Grenze (neu, offen):** `q-gem-4` („Wie viel kostet die Verlängerung
eines Grabes pro Jahrzehnt?", Gold `gemeinde`) wird von **Qwen2.5-7B-Instruct**
hart als `none` gewählt (`p(none) = 1.000`, Top-1 mit logprob −0.000). Derselbe
Prompt über Qwen2.5-Coder-7B ergibt `D` (falsch) mit p(none) ≈ 0.72,
über **gemma-4-12B** ≈ 0.08 → richtig geroutet. Kein Code-Defekt, sondern eine
SLM-Grenze (der Wortlaut „pro Jahrzehnt" ohne KB-Kontext); Test 2 bleibt für
Modelle mit dieser Antwort rot. `gemma-4-12B` erfüllt die vier Kernzusagen (S7)
vollständig, ebenso wie Coder.

**Kalibrierung (gemma, auto):** `route_accuracy 1.0`, `skip = 0.208`
(bei Coder 0.881) — der Unterschied zeigt die Modellabhängigkeit der
Auto-Schwelle; beide erfüllen S7 auf ihren Daten.
