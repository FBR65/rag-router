# SPEC: Entscheidungs-Pipeline mit wählbarem Backend je Stufe

**Status:** Entwurf zur Freigabe (noch **kein** Implementierungscode geschrieben).
**Revision 2 (2026-09-29):** Nutzer-Vorgabe „drei Entscheidungswege laya | slm |
hybrid, je eigener Weg, einfach integrierbar". **Revision 2 ersetzt §2–§5.**
§1 (Befund) bleibt gültig und ist die Begründung.
**Revision 3 (2026-09-29):** Nutzer-Antworten §8 (Endpoint wählt der Nutzer;
Logprobs automatisch; alle drei Wege; Rest „weiß nicht" → Automatik; Env
generisch). **Revision 3 ersetzt §4, §5 und §8 und ergänzt §2.3 (Automatik).**
**Datum:** 2026-09-29
**Bezug:** Integrationstest `tests/test_integration.py::TestLayaPipeline` (4 rot),
Diagnose in dieser Sitzung.

Die Spec ist append-only. Wird sie durch die Implementierung widerlegt, wird sie
sichtbar korrigiert, nie stillschweigend geändert.

---

## 1. Befund (warum das kein Kalibrierungsproblem ist)

Der Router trifft drei Entscheidungen:

- **D1 Skip** – Recherche ja/nein (`none` vs. irgendeine KB)
- **D2 KB-Wahl** – welche der N Wissensbasen
- **D3 Answer-Check** – beantworten die Passagen die Frage

Im Ist-Zustand liegen D1 und D2 in **einer** Softmax über `{KB…, none}`:
`DecisionBackend.decide(...) -> RouteDist` mit `none`-Anteil; der Router zieht
daraus nacheinander `p(none) >= skip` und die renormalisierte KB-Führung.
Zusätzlich erzwingt `decision_backend: laya | llm` **ein** Modell für D1–D3
(der Check folgt hart dem Decision-Backend, `pipeline.py:222`).

Messung mit dem echten Laya-`multilingual`-Checkpoint (32 Fixture-Fragen):

| Größe | beantwortbar | no-retrieval |
|---|---|---|
| `p(none)` min | 0.029 | 0.062 (q-no-6) |
| `p(none)` max | **0.974** (q-fir-3) | 0.957 |

- Test 1 verlangt: alle 8 no-retrieval → `none` ⟹ `skip ≤ 0.062`.
- Test 2 verlangt: keine beantwortbare → `none` ⟹ `skip > 0.974`.

**Beide Bedingungen sind unvereinbar** – unabhängig von Schwellwert oder Code.
Weitere Messungen, die das nicht retten: verbose KB-Beschreibungen
(beantwortbar bis 0.987), separate `noul`-Skipfrage (q-fir-3 `p(none)`≈0.999),
`typed-decisions`-Checkpoint (uncalibriert, ~0.2–0.4 überall). Der lokale
OpenAI-Endpunkt trennt dagegen sauber (no-retrieval `p(none)` ≥ 0.712,
beantwortbar ≤ 0.634).

**Schlussfolgerung:** Nicht *ein* Backend, sondern **je Stufe wählbar** und D1
strukturell von D2 getrennt. Die Wahl zwischen „2 Wegen" gehört auf die
Stufen-Ebene, nicht in ein einzelnes Modell.

---

## 2. Drei Wege (Revision 2)

Der Router wählt **eine** von drei Routen-Decision-Implementierungen:

| `decision_route` | Weg | Native Mechanik |
|---|---|---|
| `laya` | Laya-Checkpoint | `choice`-Frage, `probabilities` (kalibriert oder nicht) |
| `slm` | OpenAI-kompatibles SLM | Prompt + `logprobs`, Softmax über A/B/… |
| `hybrid` | beide | ein Backend primär, das andere als Prüfer/Aggregator |

Jeder Weg ist ein `DecisionRoute`-Objekt, das intern **zwei getrennte
Entscheidungen** nativ beantwortet:

```python
class DecisionRoute(Protocol):
    def skip(self, question: str) -> SkipDist: ...                 # D1
    def choose(self, question: str, rag: Mapping[str, str]) -> KbDist: ...  # D2
```

- **LayaRoute:** D1 = `noul`-Frage „braucht diese Frage Dokumente?" (ein
  `predict`-Call), D2 = `choice` über die RAG-Keys **ohne** `none` (zweiter
  `predict`-Call).
- **SlmRoute:** D1 = eigener Prompt mit `logprobs` yes/no, D2 = A/B/…-Prompt mit
  `logprobs` (jeweils `max_tokens=1`).
- **HybridRoute:** ein `primary` und ein `secondary` (`laya|slm`), Strategie
  `cascade` oder `committee` (siehe §2.1). Beide nutzen ihre nativen D1/D2.

Wichtig: D1 und D2 sind **je Weg** getrennt (je eigener Forward-Pass). Die alte
gemeinsame Softmax über `{KB…, none}` entfällt als Entscheidungsgrundlage.
Der **Answer-Check (D3)** bleibt der bestehende `AnswerChecker` (`laya|llm`,
konfigurierbar über `answer_backend`).

### 2.1 Hybrid-Strategien

- **cascade** (Default): `primary` entscheidet; liegt eine Wahrscheinlichkeit in
  der konfigurierten Grauzone `[lo, hi]`, entscheidet `secondary` und ersetzt
  das Ergebnis. Kosten nur bei Grenzfällen.
- **committee**: beide entscheiden; kombiniert wird je Stufe, per
  `hybrid.aggregate` konfigurierbar:
  - `mean` – arithmetisches Mittel der Wahrscheinlichkeiten je Route
  - `product` – Produkt, renormalisiert
  - `max` – elementweises Maximum, renormalisiert
  - `primary` – nur `primary` (falls `secondary` nur protokolliert werden soll)
  Bei committee werden beide Ergebnisse in `RouteDecision.detail` mitgeschrieben
  (jeder Weg bleibt sichtbar), auch wenn `aggregate` nur wenige nutzt.

Vorbehalt (aus §1): Laya D1 ist auf dem deutschen Set invertiert; eine
ungewichtete `mean`-Kombination kann Laya-Noise in das Ergebnis tragen. Daher
`hybrid.aggregate` ohne Default-Zwang – der Nutzer legt fest.

### 2.2 Integrationsfläche (Vorgabe: „so einfach wie möglich")

Unverändertes Nutzungsmuster, nur per Konfiguration umgeschaltet:

```python
from rag_router.config import load_config
from rag_router.pipeline import RagRouter

router = RagRouter.from_config(load_config("config.yaml"))
router.route_and_fetch("Wie lange ist die Kündigungsfrist?")
```

- **Keine neue Pflicht-Abhängigkeit**, kein neues Paket, kein Dienst außer dem
  bereits konfigurierten SLM-Endpunkt (openai-kompatibel).
- **Ein Umschalter** `decision_route: laya | slm | hybrid`; Modelle werden
  weiterhin **lazy** geladen (`preload: false` bleibt Default), damit Import und
  Start ohne GPU/Download funktionieren.
- Bestehende Host-Integrationen (`RagRouter.injected`, eigene Backends) bleiben
  unberührt; der `DecisionRoute`-Vertrag ist optional, der Alt-Adapter
  (`LegacyRoute`) hält alte `DecisionBackend`s lauffähig.
- CLI unverändert (`route`/`ask`/`index`); neue Felder sind rein Additiv.

### 2.3 Automatik statt Zahlen (`auto`)

Der Nutzer stellt **nur** den Endpoint/Modelle ein. Alles andere hat den
Default **`auto`**; das System bestimmt die Zahl selbst.

| Feld | `auto`-Regel |
|---|---|
| `thresholds.skip` | kleinster Wert, bei dem **alle** Kalibrierungsfragen der Kategorie `no_retrieval` als `none` gelten; Fallback 0.60 |
| `thresholds.fanout` | kleinster Wert, bei dem **alle** beantwortbaren Kalibrierungsfragen in der Gold-KB landen; Fallback 0.55 |
| `thresholds.answer` | trennt beantwortbare von unbeantwortbaren Kalibrierungsfragen; Fallback 0.50 |
| `hybrid.cascade_lo/hi` | `lo = P50 − 0.10`, `hi = P50 + 0.10` über die Mehrheits-/Führungs-Wahrscheinlichkeit der Kalibrierungsfragen (P50 = Median); Fallback `[0.40, 0.60]` |
| `hybrid.aggregate` | `mean` (robust gegen einen irrenden Weg), änderbar |
| `answer_backend` | der Weg mit der höchsten Trefferquote in der Messung; `auto` = Messung |
| `decision_route` | `auto` = der Weg mit der höchsten Trefferquote in der Messung |
| `slm.logprobs_mode` | `auto`: erst `logprobs` des Endpunkts; liefert er keine brauchbaren Token-Logprobs, `text`-Fallback (bestehende Mechanik) |
| `slm.model` | **kein** Auto – der Nutzer wählt Endpoint/Modell (Vorgabe 1) |

- **Messung (`calibrate`)** läuft **einmalig** beim ersten Aufruf mit
  `auto`, wenn `calibration.cache` fehlt: die Kalibrierungsfragen werden durch
  alle konfigurierten Wege geschickt, Schwellen/Backends/`decision_route`
  bestimmt und als `CalibrationProfile` gespeichert. Danach nur noch Cache.
- **Kaltstart:** schlägt die Messung fehl (Endpoint aus) oder ist
  `calibration.enabled: false`, wird jeder `auto`-Wert durch den dokumentierten
  **Fallback** ersetzt, und `RouterResult.detail["calibration"]` meldet
  `"fallback"`. Der Router läuft immer; er blockiert nie wegen der Messung.
- **Reproduzierbarkeit:** das Profil ist versioniert
  (`generator_version`, `created_at`, `sources`); bei geändertem Fragenkatalog
  oder Endpoint wird ein neues Profil erzeugt, das alte bleibt als Datei.

---

## 3. Schnittstellen

```python
# decision/base.py
@dataclass(frozen=True)
class SkipDist:
    p_recall: float            # P(Dokumentrecherche nötig); p_none = 1 - p_recall
    raw: Any = field(default=None, repr=False, compare=False)
    source: str = "probabilities"

@dataclass(frozen=True)
class KbDist:
    probabilities: Mapping[str, float]   # nur RAG-Keys, Summe ~1, kein "none"
    raw: Any = field(default=None, repr=False, compare=False)
    source: str = "probabilities"

class DecisionRoute(Protocol):
    name: str
    def skip(self, question: str) -> SkipDist: ...
    def choose(self, question: str, rag: Mapping[str, str]) -> KbDist: ...
```

- **RouterDecisionEngine** bekommt eine `DecisionRoute`; Ablauf:
  `skip = route.skip(q)`; `p_none = 1 - p_recall`; `p_none >= skip_threshold` →
  `none`. Sonst `dist = route.choose(q, rag)`; Fan-out-/Renormalisierungslogik
  unverändert auf `dist.probabilities`.
- `distribution` enthält `{**kb_dist, "none": p_none}` (CLI-/Test-kompatibel).
- `reason`/`detail` wie heute; bei hybrid zusätzlich je Weg.

Rückwärtskompatibilität: das bestehende `DecisionBackend` (ein Modell, eine
Verteilung inkl. `none`) wird von einem Adapter `LegacyRoute` auf
`DecisionRoute` abgebildet (D1 aus `p_none`, D2 aus der renormalisierten
RAG-Masse) → alte Backends und Tests bleiben nutzbar.

---

## 4. Konfigschema (additiv, abwärtskompatibel)

```yaml
router:
  decision_route: auto           # auto | laya | slm | hybrid
  answer_backend: auto           # auto | laya | slm
  laya:
    model: multilingual
  slm:
    base_url: "${RR_ROUTER_SLM_BASE_URL}"   # Endpoint waehlt der Nutzer
    api_key: "${RR_ROUTER_SLM_API_KEY}"
    model: "${RR_ROUTER_SLM_MODEL}"         # Modell waehlt der Nutzer
    logprobs_mode: auto          # auto | logprobs | text
  hybrid:
    strategy: auto               # auto | cascade | committee
    aggregate: mean              # committee: mean | product | max | primary
  calibration:
    enabled: true
    cache: ./.rag-router-calibration.json
    questions: auto              # auto = mitgelieferter Kalibrierungssatz
    max_questions: 64
  thresholds:
    skip: auto                   # auto | Zahl
    fanout: auto
    answer: auto
```

Regeln:
- `decision_route: auto` = der Weg mit der höchsten Trefferquote aus der
  Messung; `laya | slm | hybrid` erzwingt einen Weg.
- `hybrid` braucht **keine** `primary/secondary`-Angabe mehr: `auto` wählt
  automatisch die beiden (laya + slm) und die Strategie (`cascade`, wenn die
  Grauzone groß ist, sonst `committee`).
- **Altbestand** (`decision_backend: laya|llm`) bleibt gültig und wird auf
  `laya|slm` abgebildet; Konflikt mit `decision_route` → `ConfigError`.
- Unbekannte Wege/Strategien/Aggregationen → `ConfigError` mit Feldname.
- `slm` ist der neue Name für das bisherige `llm` (gleiche Mechanik); `llm`
  bleibt als Alias akzeptiert. Kein neues Paket (openai ist vorhanden).
- **Generische Env-Expansion** (`${VAR}`, bestehende Mechanik in `config.py`)
  für Endpoint/Key/Modell – kein hartcodierter Hostnamen, kein `:8080`-Default.

---

## 5. Akzeptanzkriterien (ausführbar)

**S1 – LayaRoute: D1/D2 getrennt nativ**
`LayaRoute.skip` nutzt `noul`, `LayaRoute.choose` nutzt `choice` ohne `none`
(jeweils ein eigener `predict`-Call, im Fake-Test gezählt).

**S2 – SlmRoute: D1/D2 getrennt nativ**
`SlmRoute.skip` nutzt einen yes/no-`logprobs`-Prompt, `SlmRoute.choose` einen
A/B/…-Prompt; beide `max_tokens=1`.

**S3 – KbDist ohne `none`**
`choose(...)` liefert Schlüsselraum ohne `"none"`, Summe ≈ 1.0; Fan-out und
`clear_leader` identisch zur heutigen Logik über die KB-Masse.

**S4 – Hybrid**
`strategy=cascade`: `secondary` wird **nur** in der Grauzone gerufen
(Call-Zählung); außerhalb entscheidet `primary` allein. `strategy=committee`:
beide werden gerufen; `aggregate` (`mean|product|max|primary`) kombiniert
deterministisch; `detail` enthält beide Wege.

**S5 – Rückwärtskompatibel**
Ohne neue Felder (nur `decision_backend`) identisches Verhalten; bestehende 74
Unit-Tests unverändert grün. `decision_backend: llm` und `slm` sind äquivalent.

**S6 – Validierung**
`decision_route: foo` → `ConfigError`; `hybrid` ohne konfigurierten SLM/Laya-
Weg → `ConfigError`; `hybrid.aggregate: foo` → `ConfigError`.

**S7 – Integration (Marker `integration`)**
`decision_route: auto`, `answer_backend: auto`, Schwellen `auto` erfüllt:
alle 8 no-retrieval → `["none"]`; keine beantwortbare Frage `none`;
`route_and_fetch` „answered" ≥ 12/18; falsch „answered" ≤ 3. Zusätzlich:
`decision_route: hybrid` läuft durch und protokolliert beide Wege.
(Die Zahlen wurden mit dem SLM auf `:8080` vorab verifiziert; der Test wählt den
Weg per `auto` und belegt, dass die Automatik ihn findet.)

**S8 – Automatik (`auto`)**
Gegeben eine Kalibrierungsmenge mit bekannten Kategorien: `calibrate` liefert
endliche Schwellen im Intervall `[0,1]`, `decision_route` = `slm` (der Weg mit
bester Trefferquote), ein `CalibrationProfile` mit nicht-leerer `sources`; ein
zweiter Aufruf liest den Cache (kein zweiter Modell-Call, Zählung). Endpoint
nicht erreichbar → dokumentierte Fallbacks, `detail["calibration"] == "fallback"`,
kein Absturz.

**S9 – Laya-Grenze dokumentiert**
Ein reiner Laya-Weg (`decision_route: laya`) erfüllt S7 **nicht**; README/Moduldoc
nennt die Inversion von `p(none)` auf dem deutschen Set und verweist auf
`slm`/`hybrid`/`auto`.

---

## 6. Negative Invarianten (müssen überleben)

- Öffentliche Signaturen `RagRouter.injected/from_config`, `route`,
  `route_and_fetch`, `RouterResult`, `RouteDecision` bleiben gültig
  (nur additive Parameter).
- Bestehende 74 Unit-Tests laufen **unverändert** grün.
- Keine neue Abhängigkeit (openai, laya, pydantic sind vorhanden).
- `none` bleibt reservierter RAG-Key; `RAGS`-Schema unverändert.
- Ohne neue Config-Felder identisches Verhalten wie heute.
- Fan-out-Grenzverhalten (FL-ULP, inklusiver Skip) aus `router.py` bleibt.

---

## 7. Setup-Plan

- **Neue Abhängigkeiten:** keine (nur `uv`-vorhandene Pakete).
- **Neue Dateien:** `docs/spec-decision-stages.md` (dieses Dokument, versioniert);
  Tests in bestehenden Dateien + `tests/test_decision_routes.py`.
- **Git:** Checkpoint-Commit bei Spec-Freigabe, je RED/GREEN/REFACTOR-Zyklus,
  finaler Gauntlet-Commit (nur mit Freigabe).
- **Gauntlet:** voller Loop; Integrationstest `-m integration` benötigt den
  lokalen Endpunkt (`http://127.0.0.1:8080/v1`, Modell per Env, z. B.
  `RR_ROUTER_SLM`); ohne Env/Endpoint skippt der Test mit Begründung.
- **Umgebungsvariablen:** neu `RR_ROUTER_SLM_BASE_URL`/`RR_ROUTER_SLM_API_KEY`
  (Fallback auf bestehende `OLLAMA_BASE_URL`/`OLLAMA_API_KEY`).

---

## 8. Offene Entscheidungen (bitte vor Freigabe festlegen)

Aus den Nutzerantworten festgelegt (Rev. 3):

1. **SLM-Modell/Endpoint:** vom Nutzer; **Logprobs automatisch** (`auto`-Erkennung
   + `text`-Fallback).
2. **Alle 3 Wege:** `laya | slm | hybrid`; die Messung wählt den besten
   (`decision_route: auto`).
3.–6. **Automatik:** `aggregate`, Cascade-Grauzone, `answer_backend`, `skip` –
   alle `auto` aus der Messung (mit dokumentierten Fallbacks).
7. **`TestLayaPipeline`:** wird auf `decision_route: auto` + Kalibrierung
   umgestellt; der reine Laya-Weg bleibt als dokumentierte Grenze (S9).
8. **Env:** generisch `RR_ROUTER_SLM_BASE_URL`/`RR_ROUTER_SLM_API_KEY`/
   `RR_ROUTER_SLM_MODEL` (Fallback `OLLAMA_*`); kein Default-Host.

Offen (Block für Freigabe):

1. **Kalibrierungsfragen:** Woher? Vorschlag: `tests/fixtures/questions_de.json`
   als Quelle **einbinden** (nicht duplizieren) und zusätzlich
   `calibration.questions` als eigener YAML/JSON-Pfad erlauben. Ist das ok, oder
   soll der Kalibrierungssatz von den Integrationstest-Fragen **getrennt** sein
   (dann overfittet die Messung nicht auf den Test)?
2. **`CalibrationProfile`-Ablage:** Datei `./.rag-router-calibration.json`
   (Default) ok, oder soll der Cache optional auch via Env steuerbar sein?
3. **Freigabe-Commits:** Checkpoint-Commits bei Spec-Freigabe, je GREEN/REFACTOR
   und zum Gauntlet – einverstanden?

