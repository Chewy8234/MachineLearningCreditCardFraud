#!/usr/bin/env bash
# Development/debugging: run ONE federated job through the Flower CLI (`flwr run`).
#   scripts/flwr_debug.sh kd       # heterogeneous KD (smoke config, seed 42)
#   scripts/flwr_debug.sh fedavg   # homogeneous FedAvg
#
# `flwr run` starts a local SuperLink daemon and spawns Flower/Ray processes from
# the interpreter's path; both break if that path contains whitespace. The
# experiment runner works around this automatically, but `flwr run` cannot, so
# this script needs a venv at a space-free path, e.g.:
#   uv venv --python 3.12 ~/.venvs/heterofl && uv pip install --python ~/.venvs/heterofl/bin/python -e ".[dev]"
#   PYTHON=~/.venvs/heterofl/bin/python scripts/flwr_debug.sh kd
# Equivalent without the CLI (same ServerApp/ClientApp, Simulation Runtime, works anywhere):
#   python -m heterofl.federation.simulation --job <job.json printed below>
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # robust even if the editable .pth is skipped
MODE="${1:-kd}"
PY="${PYTHON:-.venv/bin/python}"
BIN="$(cd "$(dirname "$PY")" && pwd)"
if [[ "$BIN" =~ [[:space:]] ]]; then
  echo "ERROR: interpreter path '$BIN' contains whitespace; flwr run / Ray cannot spawn" >&2
  echo "workers from it. Use a space-free venv (see header of this script), or run:" >&2
  echo "  $PY -m heterofl.experiments.runner --config configs/smoke.yaml --prepare-job $MODE" >&2
  echo "  $PY -m heterofl.federation.simulation --job <printed job.json>" >&2
  exit 2
fi
JOB="$("$PY" -m heterofl.experiments.runner --config configs/smoke.yaml --prepare-job "$MODE" \
        | sed -n 's/^Wrote //p')"
echo "Job: $JOB"
export PATH="$BIN:$PATH" FLWR_TELEMETRY_ENABLED=0
flwr run . --stream --run-config "job-path=\"$JOB\"" --federation-config "num-supernodes=3"
echo "Note: flwr run leaves a local SuperLink daemon running; stop it with: pkill -f flower-superlink"
