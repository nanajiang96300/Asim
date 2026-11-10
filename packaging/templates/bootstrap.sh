#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert sys.version_info[:2]==(3,12), "offline wheel requires Python3.12; see README for other versions"'
python3 -m venv --without-pip .venv
PYTHONPATH="$PWD/vendor/wheels/pip-26.1.2-py3-none-any.whl" .venv/bin/python -m pip install --no-index --find-links vendor/wheels pip==26.1.2 -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target Simulator -j"${ASIM_BUILD_JOBS:-1}"
printf '%s\n' 'Build complete. Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/smoke.py --operator all --output work/run1'
