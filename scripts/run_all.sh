#!/usr/bin/env bash
# PROFILE B: full research experiment (3 seeds x 4 heterogeneity levels x 5
# conditions + ablations; ~75 min on a 12-core laptop CPU), the
# random-proxy sensitivity profile, and the publication report.
# Completed runs are skipped, so re-running resumes after an interruption.
# Pass --overwrite to re-run completed runs (old results are archived, not deleted).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # robust even if the editable .pth is skipped
PY="${PYTHON:-.venv/bin/python}"
"$PY" -m heterofl.experiments.runner --config configs/paper.yaml --no-report "$@"
"$PY" -m heterofl.experiments.runner --config configs/paper_random_proxy.yaml --no-report "$@"
"$PY" scripts/generate_report.py --results results/full
"$PY" scripts/generate_report.py --results results/full_random_proxy
