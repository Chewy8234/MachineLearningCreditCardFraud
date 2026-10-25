#!/usr/bin/env bash
# PROFILE B: full research experiment (3 seeds x 4 heterogeneity levels x 5
# conditions + ablations; ~75 min on a 12-core laptop CPU), the
# random-proxy sensitivity profile, and the publication report.
# Completed runs are skipped (after verifying their identity), so re-running resumes
# after an interruption. Pass --overwrite to re-run completed runs (archived, not deleted).
# Output directories (the defaults hold LEGACY results, which the runner will not extend):
#   FULL_RESULTS=results/full_v2 PROXY_RESULTS=results/full_random_proxy_v2 scripts/run_all.sh --require-clean-source
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # robust even if the editable .pth is skipped
PY="${PYTHON:-.venv/bin/python}"
FULL="${FULL_RESULTS:-results/full}"
PROXY="${PROXY_RESULTS:-results/full_random_proxy}"
"$PY" -m heterofl.experiments.runner --config configs/paper.yaml --no-report --set "results_dir=$FULL" "$@"
"$PY" -m heterofl.experiments.runner --config configs/paper_random_proxy.yaml --no-report --set "results_dir=$PROXY" "$@"
"$PY" scripts/generate_report.py --results "$FULL"
"$PY" scripts/generate_report.py --results "$PROXY"
