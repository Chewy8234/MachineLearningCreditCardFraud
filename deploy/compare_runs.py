"""Compare two result trees run-by-run (e.g. laptop results/full vs a retrieved NRP unit).

    python deploy/compare_runs.py results/full results/nrp/nrp-validate-cpu4/unit_00_seed42_iid [--tol 1e-6]

Exact equality is NOT expected across CPU architectures (arm64 vs x86_64) or thread counts: floating-point
summation order differs. The report gives the maximum absolute difference per metric and per
prediction vector so you can judge it against the bootstrap CIs, not just pass/fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heterofl.utils.io import load_npz, read_json  # noqa: E402


def run_dirs(root: Path):
    for st in sorted(root.glob("seed_*/*/**/status.json")):
        if "_archive" not in st.parts:
            yield st.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    worst = 0.0
    n = 0
    for cand in run_dirs(a.candidate):
        rel = cand.relative_to(a.candidate)
        ref = a.reference / rel
        if not (ref / "metrics.json").is_file() or not (cand / "metrics.json").is_file():
            print(f"MISSING  {rel}")
            continue
        n += 1
        rs, cs = read_json(ref / "metrics.json")["systems"], read_json(cand / "metrics.json")["systems"]
        for system in sorted(set(rs) | set(cs)):
            if system not in rs or system not in cs:
                print(f"SYSTEM-MISMATCH {rel} {system}")
                continue
            ap_r = rs[system]["test"]["average_precision"]
            ap_c = cs[system]["test"]["average_precision"]
            pr, pc = ref / "predictions" / f"{system}.npz", cand / "predictions" / f"{system}.npz"
            dmax = float(np.max(np.abs(load_npz(pr)["p_test"] - load_npz(pc)["p_test"]))) \
                if pr.is_file() and pc.is_file() else float("nan")
            worst = max(worst, abs(ap_r - ap_c))
            flag = "ok " if abs(ap_r - ap_c) <= a.tol else "DIFF"
            print(f"{flag} {str(rel):55s} {system:28s} dAP={ap_c - ap_r:+.2e} max|dp_test|={dmax:.2e}")
    print(f"\ncompared {n} runs; worst |dAP| = {worst:.3e} (tol {a.tol:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
