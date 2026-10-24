#!/usr/bin/env python
"""Generate tables (CSV + LaTeX), figures (PNG + PDF) and summaries from results.

    python scripts/generate_report.py --results results/full
    python scripts/generate_report.py --results results/smoke
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
from heterofl.evaluation.report import generate_report
from heterofl.utils.logging import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="results directory, e.g. results/full")
    args = ap.parse_args()
    setup_logging()
    root = Path(args.results)
    if not root.is_dir():
        print(f"No such results directory: {root}", file=sys.stderr)
        return 2
    out = generate_report(root)
    print(f"Report: {out / 'summary.md'}\nTables: {root / 'tables'}\nFigures: {root / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
