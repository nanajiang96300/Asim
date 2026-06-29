#!/bin/bash
# Asim Benchmark Suite — runs a single operator and captures all metadata
# Usage: scripts/run_benchmark_suite.sh <algorithm> <mode> <config_json> [models_list_json]
set -euo pipefail

ALGORITHM="${1:?Usage: $0 <algorithm> <mode> <config> [models_list]}"
MODE="${2:?}"
CONFIG_JSON="${3:?}"
MODELS_LIST="${4:-}"

SIMULATOR="$(realpath build/bin/Simulator)"
HARDWARE_CONFIG="configs/ascend_910b_quiet.json"

# Create results directory
RESULTS_BASE="results/${ALGORITHM}"
mkdir -p "${RESULTS_BASE}"

# Determine run number
RUN_NUM=1
while [ -d "${RESULTS_BASE}/run_$(printf '%03d' ${RUN_NUM})" ]; do
  RUN_NUM=$((RUN_NUM + 1))
done
RUN_DIR="${RESULTS_BASE}/run_$(printf '%03d' ${RUN_NUM})"
mkdir -p "${RUN_DIR}"

# Capture metadata
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "${TIMESTAMP}" > "${RUN_DIR}/timestamp"

# Version info
if git rev-parse --short HEAD >/dev/null 2>&1; then
  echo "$(git rev-parse --short HEAD) ($(git diff --stat | tail -1))" > "${RUN_DIR}/version.txt"
else
  echo "unversioned-$(date +%Y%m%d-%H%M%S)" > "${RUN_DIR}/version.txt"
fi

# Copy config files
cp "${HARDWARE_CONFIG}" "${RUN_DIR}/config.json"
if [ -n "${MODELS_LIST}" ]; then
  cp "${MODELS_LIST}" "${RUN_DIR}/models_list.json"
fi

# Copy operator version docs
if [ -f "src/inverse/${ALGORITHM}/VERSION.md" ]; then
  cp "src/inverse/${ALGORITHM}/VERSION.md" "${RUN_DIR}/operator_version.md"
fi

# Run simulation
echo "[$(date)] Running ${ALGORITHM} (${MODE}) -> ${RUN_DIR}"
TRACE_CSV="${RUN_DIR}/trace.csv"
FORMULA_JSON="${RUN_DIR}/formula_steps.json"

ONNXIM_TRACE_CSV="${TRACE_CSV}" \
ONNXIM_FORMULA_JSON="${FORMULA_JSON}" \
ONNXIM_MAX_CORE_CYCLES=200000 \
"${SIMULATOR}" \
  --config "${HARDWARE_CONFIG}" \
  --models_list "${MODELS_LIST:-example/cholesky_test.json}" \
  --mode "${MODE}" \
  --log_level info \
  2>&1 | tee "${RUN_DIR}/log.txt"

# Extract summary
FINISH_CYCLE=$(grep -oP 'Current Cycle:\K\d+' "${RUN_DIR}/log.txt" | tail -1 || echo "N/A")
TOTAL_TILES=$(grep -oP 'Total tile: \K\d+' "${RUN_DIR}/log.txt" || echo "N/A")
TPS=$(grep -oP 'TPS\): \K[\d.]+' "${RUN_DIR}/log.txt" || echo "N/A")
SIM_TIME=$(grep -oP 'Simulation time: \K[\d.]+' "${RUN_DIR}/log.txt" || echo "N/A")

cat > "${RUN_DIR}/summary.json" <<EOF
{
  "algorithm": "${ALGORITHM}",
  "mode": "${MODE}",
  "timestamp": "${TIMESTAMP}",
  "run_dir": "${RUN_DIR}",
  "finish_cycle": ${FINISH_CYCLE},
  "total_tiles": ${TOTAL_TILES},
  "tps": ${TPS},
  "simulation_time_seconds": ${SIM_TIME}
}
EOF

echo ""
echo "===== ${ALGORITHM} done ====="
echo "  Finish cycle: ${FINISH_CYCLE}"
echo "  TPS: ${TPS}"
echo "  Results: ${RUN_DIR}"
