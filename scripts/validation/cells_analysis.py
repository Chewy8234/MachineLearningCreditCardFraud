"""Cell-level (scenario x partition seed) paired analysis for single-split groups (tuned, perms, stress, sweeps).

Prints mean test AP by system/scenario for each group and paired contrasts with exact sign-flip p over the
9 cells (descriptive: one outer split => test-set variance is NOT captured). Optional --ref group for deltas.
Usage: cells_analysis.py --out DIR --group name=path [--group ...] [--ref name]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from campaign_analysis import collect, COMPARISONS, cell_diffs, signflip_p, SCEN, AP


def main(groups, out, ref):
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for n, p in groups.items():
        d = collect(p)
        d["split"] = n
        data[n] = d
    lv = pd.concat([d[d.variant == "primary"].groupby(["scenario", "system"])[AP].mean().rename(n)
                    for n, d in data.items()], axis=1)
    if ref:
        for n in data:
            if n != ref:
                lv[f"{n}-{ref}"] = lv[n] - lv[ref]
    lv.to_csv(out / "levels.csv")
    rows = []
    for n, d in data.items():
        for name, spec in COMPARISONS.items():
            try:
                diffs = cell_diffs(d, name, spec)
            except Exception:
                continue
            if diffs.empty:
                continue
            for scope in SCEN + ["pooled"]:
                x = diffs if scope == "pooled" else (diffs.xs(scope, level="scenario") if scope in diffs.index.get_level_values("scenario") else None)
                if x is None or len(x) == 0:
                    continue
                rows.append({"group": n, "comparison": name, "scope": scope, "n_cells": len(x), "mean_diff": x.mean(),
                             "n_pos": int((x > 0).sum()), "p_signflip": signflip_p(x.values) if len(x) <= 18 else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(out / "contrasts.csv", index=False)
    return lv, t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    lv, t = main({g.split("=", 1)[0]: Path(g.split("=", 1)[1]) for g in a.group}, a.out, a.ref)
    pd.set_option("display.width", 250)
    print(lv.round(4).to_string())
    print(t[t.scope == "pooled"].round(4).to_string(index=False))
