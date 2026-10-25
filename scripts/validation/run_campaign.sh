#!/usr/bin/env bash
# Validation campaign driver. Each job = one runner invocation with its own results dir.
# Usage: scripts/validation/run_campaign.sh <group> [extra runner args]
#   groups: splits | perms | temporal | masktime | nodup | proxy | rounds | all
# Env: PYTHON (default ~/venvs/heterofl/bin/python), PAR (parallel jobs, default 2; each job uses
#      federation.client_num_cpus*3 cores), SPLIT_SEEDS (default "1 2 3 4 5").
# Completed runs are skipped on re-invocation (runner resume), so the script is restartable.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PY="${PYTHON:-$HOME/venvs/heterofl/bin/python}"
PAR="${PAR:-2}"
SPLIT_SEEDS="${SPLIT_SEEDS:-1 2 3 4 5}"
GROUP="${1:-all}"; shift || true
JOBS=()
add() { JOBS+=("$*"); }   # "<config> <results_dir> [--set k=v ...]"
case "$GROUP" in splits|all) for s in $SPLIT_SEEDS; do
  add "configs/val_base.yaml results/val_split_$s --set split.seed=$s"; done;; esac
case "$GROUP" in perms|all) for t in MRX MXR RMX RXM XMR; do
  add "configs/val_perm_$t.yaml results/val_perm_$t"; done;; esac
case "$GROUP" in temporal|all) add "configs/val_temporal.yaml results/val_temporal";; esac
case "$GROUP" in masktime|all) add "configs/val_masktime.yaml results/val_masktime";; esac
case "$GROUP" in nodup|all) add "configs/val_nodup.yaml results/val_nodup";; esac
case "$GROUP" in proxy|all) for t in 001 003 025 random_small; do
  add "configs/val_proxy_$t.yaml results/val_proxy_$t"; done;; esac
case "$GROUP" in rounds|all) for r in 1 3 10; do
  add "configs/val_rounds_$r.yaml results/val_rounds_$r"; done;; esac
run_job() {
  read -r cfg out rest <<<"$1"
  mkdir -p "$out"
  # shellcheck disable=SC2086
  "$PY" -m heterofl.experiments.runner --config "$cfg" --no-report --require-clean-source \
      --set "results_dir=$out" ${rest:-} > "$out.log" 2>&1
  echo "[$(date +%T)] finished $out (exit $?)"
}
export -f run_job
printf '%s\n' "${JOBS[@]}" | xargs -P "$PAR" -I{} bash -c 'run_job "{}"'
