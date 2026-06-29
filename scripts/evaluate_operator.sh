#!/bin/bash
# UOBS Phase 3 — Black-Box Operator Evaluator (CLI Wrapper)
# git stash → apply patch (or --baseline skip) → rebuild → simulate → score → stash pop

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCORER="$SCRIPT_DIR/uobs_scorer.py"
NR=64; NT=16; SNR_DB="0,5,10,15,20"; MODE=""; PATCH=""; CONFIG=""; BASELINE=false

die() { echo "{\"score\":null,\"status\":\"FAIL\",\"error\":\"$1\"}"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --patch) PATCH="$2"; shift 2;; --config) CONFIG="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;; --nr) NR="$2"; shift 2;;
    --nt) NT="$2"; shift 2;; --snr-db) SNR_DB="$2"; shift 2;;
    --baseline) BASELINE=true; shift;;
    *) die "Unknown: $1";;
  esac
done
[ -n "$CONFIG" ] && [ -n "$MODE" ] || die "Missing --config/--mode"
if [ "$BASELINE" = false ]; then
  [ -n "$PATCH" ] || die "Missing --patch (or use --baseline)"
  [ -f "$PATCH" ] || die "Patch not found"
fi
[ -f "$CONFIG" ] || die "Config not found"

cd "$ROOT"

# Save state and apply patch (skip if baseline)
if [ "$BASELINE" = false ]; then
  git stash push -- src/operations/ example/ -m "UOBS_AUTO_EVAL" 2>/dev/null || true
fi

restore() {
  cd "$ROOT"
  if [ "$BASELINE" = false ]; then
    git stash pop 2>/dev/null || git checkout -- src/operations/ example/ 2>/dev/null || true
  fi
}
trap restore EXIT

if [ "$BASELINE" = false ]; then
  # Apply patch
  git apply --whitespace=nowarn "$PATCH" 2>/tmp/_uobs_patch.err || die "PATCH_FAILED: $(head -1 /tmp/_uobs_patch.err)"
  echo "{\"status\":\"PATCH_APPLIED\"}"
fi

# Rebuild
timeout 120 cmake --build build_asim --target Simulator -j$(nproc) 2>/tmp/_uobs_build.err || \
  die "BUILD_FAILED: $(tail -3 /tmp/_uobs_build.err | tr '\n' ' ')"

# Run simulation
WD=$(mktemp -d -t asim.XXXXXX)
FJ="$WD/formula_steps.json"; TC="$WD/trace.csv"
export ONNXIM_FORMULA_JSON="$FJ" ONNXIM_TRACE_CSV="$TC"
export ONNXIM_MAX_CORE_CYCLES=100000 ONNXIM_HOME="$ROOT"
timeout 60 build_asim/bin/Simulator --config configs/ascend_910b_quiet.json \
  --models_list "$(realpath "$CONFIG")" --mode "$MODE" >/tmp/_uobs_sim.log 2>&1 || \
  die "SIM_FAILED: $(grep -i error /tmp/_uobs_sim.log | tail -1 || echo timeout)"

[ -f "$FJ" ] || die "SIM_FAILED: no formula_steps.json"
[ -f "$TC" ] || die "SIM_FAILED: no trace.csv"

# Score
python3 "$SCORER" --formula "$FJ" --trace "$TC" --nr "$NR" --nt "$NT" \
  --snr-db "$SNR_DB" --json 2>/dev/null || die "SCORE_FAILED"

rm -rf "$WD" 2>/dev/null || true
