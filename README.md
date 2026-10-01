# rag-router

A generic RAG router as a Python library. Given a question, it decides **whether
to retrieve at all** and **which of N knowledge bases (RAGs) to search**, then
retrieves evidence, checks whether the passages actually answer the question, and
reports a final state. The knowledge bases, thresholds and models are configured
in YAML; no code changes are needed to add or remove a RAG.

The routing decision itself is pluggable. Three decision routes are available
(`laya`, `slm`, `hybrid`) plus `auto`, which measures the available routes once
and derives thresholds, route and answer backend itself.

---

## Contents

- [Why split the decision](#why-split-the-decision)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Decision routes](#decision-routes)
- [Automatic calibration (`auto`)](#automatic-calibration-auto)
- [Library API](#library-api)
- [CLI](#cli)
- [Retrieval backends](#retrieval-backends)
- [Models over llama-swap](#models-over-llama-swap)
- [Testing and gauntlet](#testing-and-gauntlet)
- [Architecture](#architecture)
- [Design decisions and known limits](#design-decisions-and-known-limits)
- [License](#license)

---

## Why split the decision

The router answers **two independent questions** per request, and each decision
route answers them natively in separate forward passes:

1. **Skip (D1)** — does this question need document retrieval at all?
   `p(none) >= skip` → no search is performed.
2. **Knowledge-base choice (D2)** — among the RAGs only (never `none`):
   a clear leader `p >= fanout` is searched alone, otherwise the top-2 are
   fanned out and the answer check decides.

Measured with the real Laya `multilingual` checkpoint on the 32-question German
fixture set, `p(none)` is **inverted**: answerable questions reach up to 0.974
while the no-retrieval questions start at 0.062. Satisfying both promises
("skip every no-retrieval question" and "never skip a real question") with a
single threshold is arithmetically impossible — regardless of code or threshold
value. That finding is why the route is selectable and why D1 and D2 are
structurally separated (`docs/spec-decision-stages.md`, §1; documented
limit S9).

A single softmax over `{RAG…, none}` additionally lets a large `none` mass
displace the knowledge-base mass. Separating the stages removes that coupling.

## Requirements

- Python **3.12+**
- CPU-only is sufficient; the project deliberately pins CPU PyTorch wheels
  (see `pyproject.toml`), saving ~2.5 GB of CUDA runtime in the venv.
- Optional: an OpenAI-compatible endpoint for the `slm`/`hybrid`/`auto` routes
  and for the SLM answer check. Without one the router still runs with `laya`,
  fully offline.

Runtime dependencies: `flagembedding`, `lancedb`, `numpy`, `openai`, `pyyaml`,
`rank-bm25`, `scipy`, `transformers`, `torch`, `laya`.

## Installation

```bash
git clone <repository-url> rag-router
cd rag-router
uv sync
```

`uv sync` installs the runtime dependencies plus the `dev` group (pytest,
pytest-cov, pytest-randomly, ruff, mypy).

## Quick start

```bash
# 1. Index a knowledge base from a JSON file
uv run rag-router index --config config.yaml --rag gemeinde --texts-file gemeinde.json

# 2. Decide where a question goes (no retrieval)
uv run rag-router route --config config.yaml "Wie viele Unterschriften braucht eine Petition?"

# 3. Full pipeline: route, retrieve, answer check
uv run rag-router ask --config config.yaml --top-k 3 "Wie viel kostet das Mittagsgericht?"
```

As a library:

```python
from rag_router.config import load_config
from rag_router.pipeline import RagRouter

router = RagRouter.from_config(load_config("config.yaml"))
result = router.route_and_fetch("Wie lange ist die Rueckgabe?")

result.final              # "answered" | "not_found" | "no_retrieval"
result.hits               # evidence chunks: rag_key, doc_id, text, score
result.check.p_answered   # probability of the answer check
result.decision.routes    # e.g. ["gemeinde"] or ["firma", "dienste"] or ["none"]
```

`config.example.yaml` is a complete, runnable example. The documented index
format is `{"documents": [{"id": str, "text": str}, …]}` or
`{"texts": [str, …]}`.

## Configuration

All paths are resolved relative to the YAML file itself. `${VAR}` placeholders
are expanded from the environment; a missing variable is a hard configuration
error (no silent empty default).

```yaml
router:
  decision_route: auto            # laya | slm | hybrid | auto
  answer_backend: auto            # auto | laya | slm
  language: multilingual
  laya:
    model: multilingual           # english | multilingual | typed-decisions
    max_len: 1024                 # 512 | 1024
    preload: false                # lazy loading: import/start without download
  # Required for decision_route slm | hybrid | auto — no host or port hardcoded:
  slm:
    base_url: "${RR_ROUTER_SLM_BASE_URL}"
    api_key: "${RR_ROUTER_SLM_API_KEY}"
    model: "${RR_ROUTER_SLM_MODEL}"
  hybrid:
    strategy: auto                # auto | cascade | committee
    aggregate: mean               # committee: mean | product | max | primary
    cascade_lo: 0.40              # cascade gray zone
    cascade_hi: 0.60
  calibration:
    enabled: true
    cache: ./.rag-router-calibration.json
    questions: null               # null = bundled calibration set
    max_questions: 64
  thresholds:
    skip: auto                    # auto | number in [0, 1]
    fanout: auto
    answer: auto
    rrf_k: 60                     # RRF constant (AurumVector default)
  defaults:
    top_k: 5
    rerank: true                  # bge-reranker-v2-m3
    embed:
      model: BAAI/bge-m3
      device: cpu
      batch_size: 16
      # Optional: run bge-m3 on an OpenAI-compatible endpoint instead of
      # FlagEmbedding in-process (see "Models over llama-swap"):
      # base_url: "${RR_ROUTER_EMBED_BASE_URL}"
      # api_key: not-needed
      # `model` above then names the served model, e.g. bge-m3-gguf
rags:
  gemeinde:
    description: "Hausregeln und Verwaltung der Gemeinde Aldersbrunn"
    backend:
      type: lancedb
      db: ./data/lancedb
      table: gemeinde_chunks
    retriever: hybrid             # dense | sparse | fts | hybrid
    top_k: 3
    rerank: true
  firma:
    description: "Betriebsregeln der Firmen Haldensteg GmbH und Brunsiek Logistik"
    backend:
      type: lancedb
      db: ./data/lancedb
      table: firma_chunks
    retriever: hybrid
    top_k: 5
    rerank: false                 # example of a per-RAG override
```

Notes:

- **RAG order matters.** YAML insertion order defines the option order used by
  the Laya choice question and the A/B/C labels of the SLM prompt.
- **RAG keys** must match `^[a-z][a-z0-9_]{0,63}$`; `none` is reserved for the
  no-retrieval route and cannot be used as a key.
- Unknown fields and out-of-range values are rejected with a descriptive
  `ConfigError` instead of being silently ignored.
- `decision_backend: laya | llm` is retained as a legacy alias (`llm` == `slm`);
  it is mutually exclusive with `decision_route`.

## Decision routes

| `decision_route` | Mechanism | Notes |
|---|---|---|
| `laya` | Laya checkpoint, two separate questions (skip / choice) | Local, no endpoint required |
| `slm` | OpenAI-compatible SLM, logprobs per decision | Also separates cases where Laya errs |
| `hybrid` | Both, as `cascade` or `committee` | Best of both, configurable aggregation |
| `auto` | Measurement selects route and thresholds | Default; no tuning required |

Each route implements a `DecisionRoute` protocol with two native methods:

```python
class DecisionRoute(Protocol):
    def skip(self, question: str, rag_descriptions=None) -> SkipDist: ...   # D1
    def choose(self, question: str, rag_descriptions) -> KbDist: ...        # D2
```

- **`LayaRoute`** — D1 is a `noul` question ("does this question refer to the
  configured knowledge bases?"); D2 is a `choice` question over the RAG keys
  **without** `none`. One `predict` call each.
- **`SlmRoute`** — D1 is a choice prompt over the concrete RAG descriptions plus
  a `none` option, read via `logprobs` (softmax over the A/B/C labels); D2 is an
  A/B/… prompt over the RAGs. Both use `max_tokens=1`. Endpoints without usable
  logprobs fall back to parsing the single letter from the response
  (`probabilities_source="text"`).
- **`HybridRoute`** — one `primary` and one `secondary` route (`laya`/`slm`).
  `cascade` lets the primary decide and consults the secondary only inside the
  gray zone `[lo, hi]`; `committee` runs both and aggregates per stage using
  `mean`, `product`, `max` or `primary`. In committee mode both routes' results
  are recorded in `RouteDecision.detail` even when the aggregation ignores one.

The **answer check (D3)** is independent of the decision route and is selected
via `answer_backend: auto | laya | slm`; `auto` follows the calibrated route.
The router never blocks on failures: if the endpoint is unreachable it falls back
to the documented cold start (`skip=0.60`, `fanout=0.55`, `answer=0.50`).

## Automatic calibration (`auto`)

With `decision_route: auto` the router measures every available route against a
labelled question set once, derives thresholds from the measured distributions,
and selects the route with the highest accuracy (`calibration.py`). The result is
cached behind a **file lock**, so concurrent processes perform exactly one
measurement while the others read the finished profile.

| Field | Derivation | Fallback |
|---|---|---|
| `skip` | Separable: slightly above the highest answerable `p(none)` (robust against outliers); not separable: the no-retrieval minimum | `0.60` |
| `fanout` | Smallest value at which all answerable calibration questions still reach their gold RAG | `0.55` |
| `answer` | Midpoint between the highest unanswerable and the lowest answerable `p_answered`; not separable: the answerable minimum | `0.50` |

The bundled calibration set is generic (`needs_retrieval` only), so without a
per-question gold RAG, `fanout` and `answer` fall back to the documented values.
Point `calibration.questions` at a JSON file that carries `gold_route` and
`answerable` per question to calibrate them as well.

The profile is versioned and reproducible (`generator_version`, `created_at`,
`sources`, `fallback`), written to `calibration.cache`, and can be inspected or
checked in. The measurement costs one forward pass per question and route; it
runs once, after which the cache is reused.

## Library API

Injected construction is intended for tests and hosts that supply their own
backends and models:

```python
from rag_router.pipeline import RagRouter

router = RagRouter.injected(config, decision=my_route, backends=my_backends,
                            checker=my_checker, calibration=profile)
```

Key entry points:

- `RagRouter.from_config(config)` — real backends and real models; lazy model
  loading.
- `RagRouter.route(question)` — the routing decision only (no retrieval).
- `RagRouter.route_and_fetch(question, top_k=None)` — routing → (fan-out)
  retrieval → answer check → `RouterResult`.
- `RagRouter.index_texts(rag_key, texts, ids=None)` and
  `index_texts_from_file(rag_key, path)` — indexing convenience. Indexing is
  idempotent (upsert on the document id).
- `RagRouter.injected(...)` — full dependency injection.

`RouterResult` fields: `decision`, `hits`, `check`, `final`
(`answered` / `not_found` / `no_retrieval`), `detail`.

Custom routes follow the two-method `DecisionRoute` protocol; legacy
`DecisionBackend`s (a single distribution including `none`) keep working through
the `LegacyRoute` adapter, which is behaviourally identical to the previous
implementation.

## CLI

The CLI is a thin client over the library, intended for manual testing. Output
is JSON on stdout.

```bash
uv run rag-router index --config config.yaml --rag <key> --texts-file <file.json>
uv run rag-router route --config config.yaml "<question>"
uv run rag-router ask   --config config.yaml [--top-k N] "<question>"
```

Exit codes: `0` success, `1` error, `2` configuration missing or invalid.

## Retrieval backends

One LanceDB table per RAG.

- **Schema:** `id`, `text`, `vector` (1024-dimensional, bge-m3 dense).
- **Modes:** `dense`, `fts`, `hybrid`.
- **Hybrid:** LanceDB `RRFReranker(K=rrf_k)` over the vector and FTS rankings,
  `k=60` by default — the same fusion constant used by AurumVector. Full-text
  search uses LanceDB's native FTS (Tantivy) with a configurable language
  (`German` for the German corpus).
- **Reranking (optional):** `bge-reranker-v2-m3` via FlagEmbedding scores the RRF
  candidates and re-sorts them; enable globally via `defaults.rerank` or per RAG
  via `rags.<key>.rerank`.

bge-m3 is used for embeddings only in this release (dense vectors). The sparse
`lexical_weights` overlay is deliberately replaced by LanceDB FTS; the fusion
semantics are identical and the sparse mode remains available as an extension
point.

## Models over llama-swap

The integration test can run **all** models through a local
[llama-swap](https://github.com/mostlygeek/llama-swap) instance on `:8080` —
`bge-m3-gguf` for embeddings, any GGUF chat model for the SLM, and Laya locally:

```bash
export RR_ROUTER_EMBED_BASE_URL=http://127.0.0.1:8080/v1
export RR_ROUTER_EMBED_MODEL=bge-m3-gguf
export RR_ROUTER_SLM_BASE_URL=http://127.0.0.1:8080/v1
export RR_ROUTER_SLM_MODEL=gemma-4-12B
uv run pytest -m integration
```

With `router.defaults.embed.base_url` set, `RagRouter.from_config` uses the
`HttpEmbedder` (`POST /v1/embeddings`) instead of loading FlagEmbedding/torch in
the test process. Both paths serve the same model: the endpoint and the local
model agree to `cos = 0.9995` with identical ranking, both 1024-dimensional and
normalized. Running embeddings through the endpoint removes the bge-m3 load from
the test process and shortens the calibration measurement noticeably.

Requirements and pitfalls:

- The llama-swap entry for `bge-m3-gguf` must start `llama-server` with
  `--embeddings`. Without it, `/v1/embeddings` answers **HTTP 501**
  (`This server does not support embeddings`).
- The chat model is selected purely via `RR_ROUTER_SLM_MODEL`; no host, port or
  model name is hardcoded anywhere in the library.

## Testing and gauntlet

```bash
uv run pytest                  # 158 unit tests, no models
uv run pytest -m integration   # + real models (bge-m3, Laya; SLM env-gated)
bash scripts/gauntlet.sh       # all layers: suite, lint, mypy, coverage, mutation, integration
```

The integration tests live in `tests/test_integration.py`:

- `TestLayaPipeline` — the pure Laya route; runs offline without an endpoint,
  proves the documented limit S9, and reports the median pipeline latency.
- `TestAutoPipeline` — route `auto` after calibration: profile present, all
  no-retrieval questions skipped, no false skip, at least 12/18 answerable
  questions reach their gold RAG and are reported `answered`, at most 3
  unanswerable questions reported `answered`.
- `TestSlmRoute` — the SLM route calibrates and both routes are constructible.

SLM-dependent tests need `RR_ROUTER_SLM_BASE_URL` and `RR_ROUTER_SLM_MODEL`
(fallback `OLLAMA_BASE_URL` / `RR_ROUTER_LLM`); without them they skip with a
reason. Fixtures: `tests/fixtures/corpus_de.json` (18 documents across 3 domains)
and `questions_de.json` (32 original German questions).

Measured results (reference machine, CPU):

| Route setup | Result |
|---|---|
| Laya pipeline, bge-m3 in-process | 3 passed; median latency 0.95 s/question |
| Laya pipeline, bge-m3 via llama-swap | 3 passed; median latency 0.61 s/question |
| `auto` with `gemma-4-12B` | 10 passed |
| `auto` with `Qwen2.5-Coder-7B` | 10 passed |
| `auto` with `Qwen2.5-7B-Instruct` | 1 failed (`test_no_false_skip_on_answerable`) |

Unit suite: 158 passed, `ruff check` clean, `mypy` clean (20 source files),
mutation 9/9 killed. The complete evidence report, including the failure
analysis, is in `docs/evidence-decision-stages.md`.

## Architecture

```
src/rag_router/
  config.py            YAML -> frozen dataclasses, strict validation
  router.py            core: skip -> renormalize -> fan-out (D1/D2 orchestration)
  decision/
    base.py            protocols and DTOs (RouteDist, SkipDist, KbDist)
    route.py           LayaRoute, SlmRoute, HybridRoute, LegacyRoute
    laya.py            Laya checkpoint wrapper
    llm.py             OpenAI-compatible decision backend (legacy adapter)
  checking/            answer check (D3): base, laya, llm
  backends/
    base.py            RagBackend protocol, SearchHit
    lancedb.py         LanceDB backend: dense / FTS / hybrid + RRF + rerank
  embedding.py         BgeM3Embedder (in-process), HttpEmbedder (endpoint)
  calibration.py       threshold derivation, route/answer-backend selection
  locking.py           file lock: exactly one calibration run
  pipeline.py          RagRouter: wiring, route_and_fetch, indexing
  cli.py               route / ask / index commands
  data/                bundled calibration question set
docs/
  spec-decision-stages.md     specification (append-only, revisioned)
  evidence-decision-stages.md spec -> test mapping and gauntlet evidence
scripts/                env_check.py, verify_llm_config.py, gauntlet.sh, mutation_check.sh
tests/                  168 tests (158 unit + 10 integration), fixtures
```

## Design decisions and known limits

- **Inclusive skip comparison.** `p(none) >= skip` skips; `0.5999` searches. A
  falsely skipped real question means a wrong answer, whereas an unnecessary
  search is cheap — so the boundary is drawn conservatively.
- **Fan-out compares without epsilon.** A candidate inside the 1-ULP window below
  the threshold produces a fan-out. That is the conservative outcome: fan-out
  searches twice but never skips incorrectly.
- **No blocking on failures.** A missing measurement, an unreachable endpoint or
  a failed calibration never prevents the router from starting; it falls back to
  the documented cold start.
- **Laya alone does not satisfy the four core promises** on the German set
  (limit S9, measured above). `slm`, `hybrid` and `auto` do.
- **The route depends on the SLM's capability.** With `Qwen2.5-7B-Instruct`, the
  answerable question `q-gem-4` (grave extension fee) is classified as `none`
  with `p(none) = 1.000` (top-1 logprob −0.000), which fails
  `test_no_false_skip_on_answerable`. The same prompt yields a wrong but
  non-skipping `D` (p(none) ≈ 0.72) on `Qwen2.5-Coder-7B` and a correct route
  (p(none) ≈ 0.08) on `gemma-4-12B`. This is a model limit, not a code defect,
  and the test was deliberately not weakened.
- **Thresholds are model-dependent.** Calibrated `skip` was `0.881` with
  Coder-7B and `0.208` with gemma-4-12B; both satisfy the promises on their own
  calibration data. Re-run calibration after switching models.
- **`_run_calibration` is expensive** (one forward pass per question and route).
  The file lock prevents duplicate parallel runs but does not make a single run
  faster.
- **No property-based testing framework.** The property tests are a stdlib
  random sweep, weaker than a real PBT tool (no automatic shrinking).
- **Sparse retrieval is delegated to FTS.** See
  [Retrieval backends](#retrieval-backends).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Frank Reis.
