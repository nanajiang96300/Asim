#!/bin/bash
set -euo pipefail
cd /home/nanajiang/asim-worktrees/bri-fixed-hb-20261109
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/nanajiang/Asim/.venv/bin/python
CONTRACT=/home/nanajiang/Asim/experiments/fixed_hb_native/2026-11-09-v2/contract.json
DIGEST=42242c3e0e52a93023b12163332a265f10b436a8acefffe337d3d9393a30f959
OUT=/home/nanajiang/Asim/results/fixed_hb_native/2026-11-09-v2
for STAGE in asim dataset replay; do
 /usr/bin/time -v "$PY" -m verification.reconstruction.stable_hb_campaign --contract "$CONTRACT" --sha256 "$DIGEST" --output "$OUT" --stage "$STAGE" > "/home/nanajiang/Asim/results/fixed_hb/native-v2-${STAGE}.log" 2>&1
done
"$PY" /home/nanajiang/Asim/experiments/fixed_hb_native/2026-11-09-v2/tools/controls.py > /home/nanajiang/Asim/results/fixed_hb/native-v2-controls.log 2>&1
"$PY" /home/nanajiang/Asim/experiments/fixed_hb_native/2026-11-09-v2/tools/regression.py > /home/nanajiang/Asim/results/fixed_hb/native-v2-regression.log 2>&1
