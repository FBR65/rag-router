# EVIDENCE: Entscheidungswege (laya | slm | hybrid | auto)

**Stand:** 2026-09-29 · **Source-State:** Commit `ee187b0` (Arbeitsbaum danach
unverändert) · **Spec:** `docs/spec-decision-stages.md` (Rev. 3, freigegeben;
Spec-Freigabe erteilt: Nutzerantworten §8 = 1 getrennt, 2 Default, 3 ok, plus
„Go bis zum Ende").

## Umfang (Tier)

**Tier 3 (hoch)** nach Kalibrierung: neue öffentliche Schnittstellen, Routing-
Entscheidung (Kern des Produkts), Automatik mit Persistenz. Failure-Model:
falsches Skip einer echten Frage, stiller Fallback, unerreichbarer Endpoint,
`auto`-Werte außerhalb [0,1], Kalibrierungs-Overfit.

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
| S8 Automatik | `test_calibration.py` (18), `test_pipeline.py::TestAutoCalibration` |
| S9 Laya-Grenze | `TestLayaPipeline::test_documented_laya_skip_inversion` |

Alle Szenarien der Spec sind auf grüne Tests abgebildet; keine Lücke.

## Gauntlet-Layer (frischer Lauf nach letzter Code-Änderung, Commit ee187b0)

| Layer | Kommando | Ergebnis |
|---|---|---|
| Volle Suite (Unit) | `uv run pytest -q` | **139 passed**, 10 deselected |
| Statische Typen | — | **übersprungen:** kein mypy/pyright im Projekt/Setup (Spec §7 nennt keins) |
| Lint/Format | `uv run ruff check src/ tests/` | **All checks passed!** |
| Coverage (Unit+Integration) | `uv run pytest -m "" --cov=rag_router` | gesamt **90 %**; neue Module: `calibration` 99 %, `decision/route` 90 %, `router` 88 %, `pipeline` 81 %, `config` 87 % |
| Integration (echtes Modell) | `RR_ROUTER_SLM_* uv run pytest -m integration` | **10 passed** (Laya offline + SLM/auto am Endpoint) |
| Mutation (manuell, 7 Mutanten) | `bash scripts/mutation_check.sh` | **7/7 getötet** |
| Property-Tests | `test_calibration.py::TestPropertiesSweep` | 2 Properties, 400 Zufalls-Eingaben; `hypothesis` nicht installiert (keine neue Abhängigkeit), stdlib-Sweep als Ersatz |
| Complexity | Sichtprüfung | neue Funktionen klein/ein Zweck; `SlmRoute`/`HybridRoute` je < 120 Zeilen |
| Real Execution | siehe unten | CLI `index`+`route`+`ask` echt am Endpoint |
| Supply chain | `pyproject.toml` | **keine neue Abhängigkeit** (openai, laya vorhanden); kein Secret im Diff |
| Suite-Health | 3× randomisiert (`pytest-randomly`) | 3× **139 passed**, keine Flakes |

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
- **Typen-Layer fehlt** (kein Typechecker im Projekt); nicht nachgerüstet, um
  keine Abhängigkeit ohne Spec-Freigabe einzuführen.
- **`hypothesis` fehlt**; Properties als stdlib-Sweep, schwächer als ein
  echtes PBT-Tool.
- **`pipeline._run_calibration` ist teuer** (ein Forward-Pass je Frage und Weg)
  und läuft nur einmalig; Cache-Pfad getestet, aber kein Nebenläufigkeits-Schutz
  (zwei Prozesse könnten parallel kalibrieren — letzter Schreibvorgang gewinnt).
- Der Kalibrierungssatz ist **generisch** (nur `needs_retrieval`), daher
  kalibriert `fanout`/`answer` auf Fallback, wenn kein KB-Gold vorliegt.

## Reproduzierbarkeit

- Dev-Versionen: pytest 9.1.1, pytest-randomly 5.0.0, pytest-cov 7.1.0,
  ruff 0.16.9 (aus `pyproject.toml`).
- Mutationsskript persistiert: `scripts/mutation_check.sh` (führt die 7 Mutanten
  aus, stellt den Arbeitsbaum nach jedem Mutanten wieder her).
- Ein Einstiegskommando (siehe `scripts/gauntlet.sh`).
- Ohne Endpoint: `uv run pytest -m "not integration"` (139) und
  `uv run pytest -m integration` skippt die SLM-Tests mit Begründung.
