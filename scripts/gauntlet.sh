#!/usr/bin/env bash
# Ein Einstiegspunkt für alle Gauntlet-Layer (docs/evidence-decision-stages.md).
# Integrationstests laufen nur, wenn RR_ROUTER_SLM_BASE_URL gesetzt ist.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/6 Unit-Suite (randomisiert) =="
uv run pytest -q

echo "== 2/6 Lint =="
uv run ruff check src/ tests/

echo "== 3/6 Statische Typen =="
uv run mypy

echo "== 4/6 Coverage (Unit) =="
uv run pytest -q --cov=rag_router --cov-report=term-missing

echo "== 5/6 Mutation (7 Mutanten) =="
bash scripts/mutation_check.sh

echo "== 6/6 Integration (nur mit Endpoint) =="
if [ -n "${RR_ROUTER_SLM_BASE_URL:-}" ]; then
  uv run pytest -m integration -q
else
  echo "uebersprungen: RR_ROUTER_SLM_BASE_URL nicht gesetzt"
fi
