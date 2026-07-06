#!/usr/bin/env bash
# CI Gate — automated verification pipeline for Asim operator correctness
# Usage: ./scripts/ci_gate.sh [--fast] [--layer <layer>] [--operator <name>]
#
#   --fast        Skip simulator runtime checks, only build + unit tests
#   --layer 1|2|3 Run only the specified verification layer
#   --operator X  Run verification only for operator X
#   --help        Show this help
#
# Layers:
#   Layer 1 (FAST): Build + unit tests + DAG executor self-test
#   Layer 2 (FULL): Layer 1 + per-operator DAG numerical verification
#   Layer 3 (DEEP): Layer 2 + trace audit + formula-trace consistency
#
# Exit codes:
#   0 — all checks pass
#   1 — build failure
#   2 — unit test failure
#   3 — DAG verification failure
#   4 — trace audit failure

set -euo pipefail
IFS=$'\n\t'

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
CONFIG="${PROJECT_ROOT}/configs/ascend_910b_quiet.json"
FORMULA_DIR="/tmp/ci_gate_formulas"
RESULTS_DIR="${PROJECT_ROOT}/results/ci_gate"

# Operator → test model mapping
declare -A OPERATOR_MODELS=(
    ["cholesky_noblock"]="cholesky_noblock_v2_test.json"
    ["cholesky_block"]="cholesky_block_v3_test.json"
    ["ldl_noblock"]="ldl_noblock_v2_test.json"
    ["ldl_block"]="ldl_block_v3_test.json"
    ["newton_schulz"]="newton_schulz_v3_test.json"
    ["bri"]="bri_v3_test.json"
)

# Operator → verify script mapping
declare -A VERIFY_SCRIPTS=(
    ["cholesky_noblock"]="cholesky_noblock_v2.py"
    ["cholesky_block"]="cholesky_block_v3.py"
    ["ldl_noblock"]="ldl_noblock_v2.py"
    ["ldl_block"]="ldl_block_v3.py"
    ["newton_schulz"]="newton_schulz_v3.py"
    ["bri"]="bri_v3.py"
)

# ── Helpers ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
TOTAL=0

pass() { echo -e "  ${GREEN}✓ PASS${NC} $1"; PASSED=$((PASSED + 1)); TOTAL=$((TOTAL + 1)); }
fail() { echo -e "  ${RED}✗ FAIL${NC} $1"; FAILED=$((FAILED + 1)); TOTAL=$((TOTAL + 1)); }
skip() { echo -e "  ${YELLOW}○ SKIP${NC} $1"; }
info() { echo -e "  ${YELLOW}→${NC} $1"; }
section() { echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $1"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# ── CLI Parsing ────────────────────────────────────────────────────────────
FAST_MODE=false
TARGET_LAYER=""
TARGET_OPERATOR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast) FAST_MODE=true; shift ;;
        --layer) TARGET_LAYER="$2"; shift 2 ;;
        --operator) TARGET_OPERATOR="$2"; shift 2 ;;
        --help) head -15 "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; head -15 "$0"; exit 1 ;;
    esac
done

# ── Pre-flight ─────────────────────────────────────────────────────────────
mkdir -p "$FORMULA_DIR" "$RESULTS_DIR"

cd "$PROJECT_ROOT"

# Activate venv if available
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate" 2>/dev/null || true
fi

# ── Layer 1: Build + Unit Tests + DAG Self-Test ───────────────────────────
run_layer1() {
    section "Layer 1: Build & Unit Tests"

    # 1a. Build Simulator
    info "Building Simulator..."
    if cmake --build "$BUILD_DIR" --target Simulator -j"$(nproc)" 2>&1 | tail -5; then
        pass "Simulator builds successfully"
    else
        fail "Simulator build failed"
        return 1
    fi

    # 1b. Build Simulator_test
    info "Building Simulator_test..."
    if cmake --build "$BUILD_DIR" --target Simulator_test -j"$(nproc)" 2>&1 | tail -5; then
        pass "Simulator_test builds successfully"
    else
        fail "Simulator_test build failed"
        return 1
    fi

    # 1c. Run GTest unit tests
    if [ -x "${BUILD_DIR}/bin/Simulator_test" ]; then
        info "Running unit tests..."
        if "${BUILD_DIR}/bin/Simulator_test" 2>&1 | tail -20; then
            pass "Unit tests pass"
        else
            fail "Unit tests failed"
            return 1
        fi
    else
        skip "Simulator_test binary not found at ${BUILD_DIR}/bin/Simulator_test"
    fi

    # 1d. DAG executor Python self-test
    info "DAG executor self-test..."
    if python3 "${PROJECT_ROOT}/scripts/uobs_dag_executor.py" 2>&1 | grep -q "Execution complete"; then
        pass "DAG executor self-test passes"
    else
        fail "DAG executor self-test failed"
        return 1
    fi
}

# ── Layer 2: Per-Operator DAG Numerical Verification ───────────────────────
run_layer2() {
    section "Layer 2: DAG Numerical Verification"

    local sim_bin="${BUILD_DIR}/bin/Simulator"
    if [ ! -x "$sim_bin" ]; then
        fail "Simulator binary not found. Run Layer 1 first."
        return 1
    fi

    local operators=("${!OPERATOR_MODELS[@]}")
    if [ -n "$TARGET_OPERATOR" ]; then
        operators=("$TARGET_OPERATOR")
    fi

    for op in "${operators[@]}"; do
        local model_file="${PROJECT_ROOT}/example/${OPERATOR_MODELS[$op]}"
        local verify_script="${PROJECT_ROOT}/scripts/verify/${VERIFY_SCRIPTS[$op]}"
        local formula_file="${FORMULA_DIR}/${op}_formula.json"

        echo ""
        info "Verifying $op..."

        # Check prerequisites
        if [ ! -f "$model_file" ]; then
            skip "$op: model file not found ($model_file)"
            continue
        fi
        if [ ! -f "$verify_script" ]; then
            skip "$op: verify script not found ($verify_script)"
            continue
        fi

        # Step 2a: Run simulator to generate formula_steps.json
        info "  Running simulator for formula output..."
        local mode=""
        case $op in
            cholesky_noblock) mode="cholesky_noblock_baseline" ;;
            cholesky_block)   mode="cholesky_block_baseline" ;;
            ldl_noblock)      mode="ldl_noblock_baseline" ;;
            ldl_block)        mode="ldl_block_baseline" ;;
            newton_schulz)    mode="newton_schulz_baseline" ;;
            bri)              mode="bri_baseline" ;;
        esac

        if ONNXIM_FORMULA_JSON="$formula_file" ONNXIM_TRACE_CSV="${FORMULA_DIR}/${op}_trace.csv" \
           timeout 120 "$sim_bin" \
           --config "$CONFIG" \
           --models_list "$model_file" \
           --mode "$mode" \
           --log_level error 2>&1 | tail -3; then
            info "  Simulator finished"
        else
            fail "$op: simulator runtime error"
            continue
        fi

        if [ ! -f "$formula_file" ] || [ ! -s "$formula_file" ]; then
            fail "$op: formula_steps.json not generated"
            continue
        fi

        # Step 2b: Run DAG verification
        info "  Running DAG verification..."
        local result
        result=$(python3 "$verify_script" "$formula_file" 2>&1) || true
        echo "  $result"

        if echo "$result" | grep -q "PASS"; then
            pass "$op: DAG numerical verification"
        else
            fail "$op: DAG numerical verification"
        fi
    done
}

# ── Layer 3: Trace Audit + Formula-Trace Consistency ──────────────────────
run_layer3() {
    section "Layer 3: Trace Audit & Consistency"

    local operators=("${!OPERATOR_MODELS[@]}")
    if [ -n "$TARGET_OPERATOR" ]; then
        operators=("$TARGET_OPERATOR")
    fi

    for op in "${operators[@]}"; do
        local formula_file="${FORMULA_DIR}/${op}_formula.json"
        local trace_file="${FORMULA_DIR}/${op}_trace.csv"

        if [ ! -f "$formula_file" ]; then
            skip "$op: formula file not found"
            continue
        fi

        # 3a. Trace audit (GEMM coverage)
        if [ -f "$trace_file" ] && [ -f "${PROJECT_ROOT}/scripts/trace_audit.py" ]; then
            info "Trace audit for $op..."
            if python3 "${PROJECT_ROOT}/scripts/trace_audit.py" "$trace_file" "$formula_file" 2>&1; then
                pass "$op: trace audit"
            else
                fail "$op: trace audit"
            fi
        else
            skip "$op: trace audit skipped (no trace file or audit script)"
        fi

        # 3b. Formula-trace consistency (GEMM coverage >= 50%)
        if [ -f "${PROJECT_ROOT}/scripts/uobs_scorer.py" ] && [ -f "$formula_file" ] && [ -f "$trace_file" ]; then
            info "Formula-trace consistency for $op..."
            if python3 "${PROJECT_ROOT}/scripts/uobs_scorer.py" "$formula_file" "$trace_file" 2>&1; then
                pass "$op: formula-trace consistency"
            else
                fail "$op: formula-trace consistency"
            fi
        fi
    done
}

# ── Summary ────────────────────────────────────────────────────────────────
summary() {
    section "CI Gate Summary"
    echo "  Total checks: $TOTAL | ${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC}"

    # Archive results
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local summary_file="${RESULTS_DIR}/ci_summary_${timestamp}.txt"
    {
        echo "CI Gate Run: $(date)"
        echo "Total: $TOTAL | Passed: $PASSED | Failed: $FAILED"
        echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    } > "$summary_file"
    info "Results archived to $summary_file"

    if [ "$FAILED" -gt 0 ]; then
        echo -e "\n${RED}CI GATE FAILED — $FAILED check(s) failed${NC}"
        return 4
    else
        echo -e "\n${GREEN}CI GATE PASSED — all $TOTAL checks passed${NC}"
        return 0
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────

if [ "$FAST_MODE" = true ]; then
    run_layer1 || exit $?
    summary
    exit $?
fi

case "$TARGET_LAYER" in
    1)
        run_layer1 || exit $?
        ;;
    2)
        run_layer1 || exit $?
        run_layer2 || true  # Don't exit on layer 2 failures, collect all results
        ;;
    3)
        run_layer1 || exit $?
        run_layer2 || true
        run_layer3 || true
        ;;
    "")
        # Full pipeline
        run_layer1 || exit $?
        if [ "$FAST_MODE" != true ]; then
            run_layer2 || true
            run_layer3 || true
        fi
        ;;
    *)
        echo "Invalid layer: $TARGET_LAYER (must be 1, 2, or 3)"
        exit 1
        ;;
esac

summary
