#!/usr/bin/env bash
# Manuelle Mutationsprüfung für die Entscheidungswege.
# Bringt je Mutant genau eine Verhaltensänderung an, erwartet, dass die Suite
# ihn tötet, und stellt den Arbeitsbaum danach wieder her. Kein Zufall, kein
# Zustand bleibt zurück. Siehe docs/evidence-decision-stages.md.
set -u
cd "$(dirname "$0")/.."

ROUTE=src/rag_router/decision/route.py
CAL=src/rag_router/calibration.py
ROUTER=src/rag_router/router.py
LOCK=src/rag_router/locking.py
TESTS=(tests/test_decision_routes.py tests/test_calibration.py tests/test_router.py tests/test_pipeline.py tests/test_locking.py)

suite_kills() {
  uv run pytest "${TESTS[@]}" -q >/tmp/mutation_check.out 2>&1
  grep -qE "[0-9]+ (failed|error)" /tmp/mutation_check.out
}

run_mutant() {
  local name="$1" file="$2" old="$3" new="$4"
  cp "$file" "$file.bak"
  python3 - "$file" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path, encoding="utf-8").read()
if old not in text:
    sys.exit(f"Mutant-Muster nicht gefunden: {old!r}")
open(path, "w", encoding="utf-8").write(text.replace(old, new, 1))
PY
  if suite_kills; then
    echo "KILLED   $name"
  else
    echo "SURVIVED $name"
  fi
  mv "$file.bak" "$file"
}

run_mutant laya_p_recall "$ROUTE" \
  "p_recall = 1.0 - float(raw)" "p_recall = float(raw)"
run_mutant renorm_skip_none "$ROUTE" \
  "total = sum(float(v) for v in probabilities.values())" \
  "total = sum(float(v) for v in probabilities.values()) + 1.0"
run_mutant skip_thr_mid "$CAL" \
  "return max_ans + 0.10 * (min_no - max_ans)" "return (max_ans + min_no) / 2.0"
run_mutant skip_thr_le "$CAL" \
  "return max_ans + 0.10 * (min_no - max_ans)" "return min_no"
run_mutant hybrid_gray "$ROUTE" \
  "return self._cascade_lo <= value <= self._cascade_hi" \
  "return self._cascade_lo < value < self._cascade_hi"
run_mutant router_skip_ge "$ROUTER" \
  "if p_none >= self._thresholds.skip:" "if p_none > self._thresholds.skip:"
run_mutant fanout_lt "$ROUTER" \
  "if ranked[0][1] < self._thresholds.fanout and len(ranked) >= 2:" \
  "if ranked[0][1] <= self._thresholds.fanout and len(ranked) >= 2:"
run_mutant lock_stale_removed "$LOCK" \
  "or time.time() - path.stat().st_mtime > stale_after" \
  "or False"
run_mutant lock_ignore_excl "$LOCK" \
  "fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)" \
  "fd = os.open(path, os.O_CREAT | os.O_WRONLY)"
