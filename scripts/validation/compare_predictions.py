"""Equivalence gate: compare per-transaction predictions of two results trees run-by-run.

    compare_predictions.py --a results/serial_ref/val_split_1 --b <run-tree>/results/v1/split_1/unit_s42_iid

Pairs every ``<seed>/<scenario>/<method>[/ablations/<variant>]/predictions/<system>.npz`` present in both
trees (any directory depth; a unit tree and a serial tree have the same layout below the results root) and
reports bitwise equality, max |dp| and the test-AP difference. Exit code 1 if any pair differs by more
than --tol in max |dp|.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def index(root: Path) -> dict[str, Path]:
    out = {}
    for f in root.rglob("predictions/*.npz"):
        rel = f.relative_to(root)
        parts = rel.parts
        if "_archive" in parts:
            continue
        out["/".join(parts)] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    A, B = index(a.a), index(a.b)
    common = sorted(set(A) & set(B))
    if not common:
        print("no common prediction files", file=sys.stderr)
        return 2
    bad, bit = 0, 0
    rows = []
    for k in common:
        za, zb = np.load(A[k]), np.load(B[k])
        dv = float(np.max(np.abs(za["p_val"] - zb["p_val"])))
        dt = float(np.max(np.abs(za["p_test"] - zb["p_test"])))
        same = bool(np.array_equal(za["p_test"], zb["p_test"]) and np.array_equal(za["p_val"], zb["p_val"]))
        ap_a = average_precision_score(za["y_test"], za["p_test"])
        ap_b = average_precision_score(zb["y_test"], zb["p_test"])
        bit += same
        bad += max(dv, dt) > a.tol
        rows.append((k, same, max(dv, dt), ap_b - ap_a))
    for k, same, d, dap in rows:
        print(f"{'==' if same else '!='} {k:<80} max|dp|={d:.2e}  dAP={dap:+.6f}")
    print(f"\n{len(common)} common prediction files; {bit} bitwise identical; {bad} exceed tol={a.tol:g}; "
          f"only in A: {len(set(A) - set(B))}, only in B: {len(set(B) - set(A))}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
