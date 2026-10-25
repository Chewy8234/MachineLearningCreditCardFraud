#!/usr/bin/env bash
# Homogeneous baselines + the heterogeneous reference, all from the SAME source revision and
# environment (the cross-profile comparison refuses to mix library versions), then the
# per-profile reports and the paired cross-profile comparison.
#
#   scripts/run_homogeneous.sh [runner flags, e.g. --require-clean-source]
#
# Completed runs are skipped after their identity is verified, so re-running resumes.
# Environment variables (all optional):
#   RESULTS_ROOT  parent of the four results directories          (default: results)
#   LANES         1 = profiles one after another; 2 = two at a time (default: 1)
#                 each profile uses 3 banks x federation.client_num_cpus = 12 cores
#   RUNNER_SETS   extra --set overrides for non-semantic locations, e.g.
#                 "data.path=/data/creditcard.csv"
#   CACHE_ROOT    if set, each lane gets its own data.processed_dir / data.partitions_dir
#                 under it (the caches are not locked; required when LANES=2). They are
#                 rebuilt deterministically; the comparison verifies the partition hashes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-.venv/bin/python}"
ROOT="${RESULTS_ROOT:-results}"
LANES="${LANES:-1}"
SETS=()
for s in ${RUNNER_SETS:-}; do SETS+=(--set "$s"); done

run() {  # run <lane> <config> <results dir> [runner flags]
  local cache=()
  if [ -n "${CACHE_ROOT:-}" ]; then
    cache=(--set "data.processed_dir=$CACHE_ROOT/lane$1/processed"
           --set "data.partitions_dir=$CACHE_ROOT/lane$1/partitions")
  fi
  "$PY" -m heterofl.experiments.runner --config "$2" --no-report \
        --set "results_dir=$3" ${SETS[@]+"${SETS[@]}"} "${cache[@]+"${cache[@]}"}" "${@:4}"
}

if [ "$LANES" = "2" ] && [ -z "${CACHE_ROOT:-}" ]; then
  echo "LANES=2 needs CACHE_ROOT (separate data caches per lane)" >&2
  exit 2
fi

# a failed run makes the runner exit 1; later profiles still run, the status is kept
lane() {  # lane <n> <config> <dir> [<config> <dir> ...] -- [runner flags]
  local n=$1 rc=0; shift
  local jobs=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do jobs+=("$1" "$2"); shift 2; done
  [ "${1:-}" = "--" ] && shift
  for ((i = 0; i < ${#jobs[@]}; i += 2)); do
    run "$n" "${jobs[i]}" "${jobs[i+1]}" "$@" || rc=1
  done
  return $rc
}

status=0
if [ "$LANES" = "2" ]; then
  # the two longest profiles (RF committee, heterogeneous + ablations) go to different lanes
  lane 1 configs/homog_random_forest.yaml "$ROOT/homog_random_forest" \
         configs/homog_mlp.yaml "$ROOT/homog_mlp" -- "$@" &
  L1=$!
  lane 2 configs/paper.yaml "$ROOT/full_v2" \
         configs/homog_xgboost.yaml "$ROOT/homog_xgboost" -- "$@" &
  L2=$!
  wait $L1 || status=1
  wait $L2 || status=1
else
  lane 1 configs/paper.yaml "$ROOT/full_v2" configs/homog_xgboost.yaml "$ROOT/homog_xgboost" \
         configs/homog_random_forest.yaml "$ROOT/homog_random_forest" \
         configs/homog_mlp.yaml "$ROOT/homog_mlp" -- "$@" || status=1
fi
[ "$status" = 0 ] || echo "Some runs failed (see each results dir's outcomes_*.csv); reporting the completed ones" >&2

for d in full_v2 homog_xgboost homog_random_forest homog_mlp; do
  "$PY" scripts/generate_report.py --results "$ROOT/$d"
done
"$PY" scripts/compare_committees.py --hetero "$ROOT/full_v2" \
      --homog-xgboost "$ROOT/homog_xgboost" --homog-random-forest "$ROOT/homog_random_forest" \
      --homog-mlp "$ROOT/homog_mlp" --out "$ROOT/committee_comparison"
exit $status
