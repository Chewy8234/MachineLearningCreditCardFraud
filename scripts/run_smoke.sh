#!/usr/bin/env bash
# PROFILE A: smoke test (development results only; ~2 min on a laptop CPU).
# Runs all five experimental conditions + ablations on a 20% stratified subsample
# of the real ULB data, then writes tables/figures to the results directory.
# results/smoke holds LEGACY results the runner will not extend, so pick another one:
#   SMOKE_RESULTS=results/smoke_v2 scripts/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # robust even if the editable .pth is skipped
PY="${PYTHON:-.venv/bin/python}"
"$PY" -m heterofl.experiments.runner --config configs/smoke.yaml \
    --set "results_dir=${SMOKE_RESULTS:-results/smoke}" "$@"
