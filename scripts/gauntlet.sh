#!/usr/bin/env bash
# Ein Einstiegspunkt für alle Gauntlet-Layer (docs/evidence-decision-stages.md).
# Integrationstests laufen nur, wenn RR_ROUTER_SLM_BASE_URL gesetzt ist.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 Unit-Suite (randomisiert) =="
uv run pytest -q

echo "== 2/5 Lint =="
uv run ruff check src/ tests/

echo "== 3/5 Coverage (Unit) =="
uv run pytest -q --cov=rag_router --cov-report=term-missing

echo "== 4/5 Mutation (7 Mutanten) =="
bash scripts/mutation_check.sh

echo "== 5/5 Integration (nur mit Endpoint) =="
if [ -n "${RR_ROUTER_SLM_BASE_URL:-}" ]; then
  uv run pytest -m integration -q
else
  echo "uebersprungen: RR_ROUTER_SLM_BASE_URL nicht gesetzt"
fi
