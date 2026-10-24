#!/usr/bin/env bash
# PROFILE A: smoke test (development results only; ~1-2 min on a laptop CPU).
# Runs all five experimental conditions + ablations on a 20% stratified subsample
# of the real ULB data, then writes tables/figures to results/smoke/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # robust even if the editable .pth is skipped
PY="${PYTHON:-.venv/bin/python}"
"$PY" -m heterofl.experiments.runner --config configs/smoke.yaml "$@"
