#!/usr/bin/env python
"""Compare the heterogeneous committee with the homogeneous baselines (paired, across profiles).

    python scripts/compare_committees.py --hetero results/full_v2 \\
        --homog-xgboost results/homog_xgboost --homog-random-forest results/homog_random_forest \\
        --homog-mlp results/homog_mlp --out results/committee_comparison

Refuses to compare directories whose dataset, splits, partitions, test labels, library
versions or configuration (other than the client model families) differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import heterofl  # noqa: F401  (must precede torch: fixes the OpenMP load order)
except ModuleNotFoundError:  # editable-install .pth skipped (e.g. macOS "hidden" flag)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import heterofl  # noqa: F401
from heterofl.evaluation.committee_comparison import ComparisonError, compare
from heterofl.utils.logging import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hetero", required=True, help="heterogeneous profile (configs/paper.yaml)")
    ap.add_argument("--homog-xgboost", help="results of configs/homog_xgboost.yaml")
    ap.add_argument("--homog-random-forest", help="results of configs/homog_random_forest.yaml")
    ap.add_argument("--homog-mlp", help="results of configs/homog_mlp.yaml")
    ap.add_argument("--out", default="results/committee_comparison")
    args = ap.parse_args()
    setup_logging()
    dirs = {"hetero": args.hetero, "xgboost": args.homog_xgboost,
            "random_forest": args.homog_random_forest, "mlp": args.homog_mlp}
    dirs = {k: Path(v) for k, v in dirs.items() if v}
    missing = [str(v) for v in dirs.values() if not v.is_dir()]
    if missing:
        print(f"No such results directory: {missing}", file=sys.stderr)
        return 2
    try:
        summary = compare(dirs, Path(args.out))
    except ComparisonError as exc:
        print(f"Refusing to compare: {exc}", file=sys.stderr)
        return 4
    print(f"Summary: {summary}\nTables: {Path(args.out) / 'tables'}\n"
          f"Figures: {Path(args.out) / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
