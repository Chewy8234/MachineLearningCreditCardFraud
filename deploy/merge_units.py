"""Merge retrieved NRP unit directories into one results tree the report generator understands.

    python deploy/merge_units.py results/nrp/nrp-paper-cpu12-v1 --out results/full_nrp_cpu12

Refuses to overwrite: a (seed, scenario) already present in --out aborts the merge (no duplicates).
Per-unit hardware.json / ledger.jsonl / runner.log / config.yaml / environment.json are kept under
<out>/_units/<unit>/ so every result stays traceable to its pod and CPU setting.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    units = sorted(a.tag_dir.glob("unit_*"))
    if not units:
        print(f"No unit_* directories in {a.tag_dir}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    for u in units:
        for seed_dir in sorted(u.glob("seed_*")):
            for scen in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                dest = a.out / seed_dir.name / scen.name
                if dest.exists():
                    print(f"Duplicate {dest}; aborting", file=sys.stderr)
                    return 1
                shutil.copytree(scen, dest)
        meta = a.out / "_units" / u.name
        meta.mkdir(parents=True, exist_ok=True)
        for f in ("hardware.json", "ledger.jsonl", "runner.log", "config.yaml", "environment.json",
                  "dataset_audit.json", "split_manifest.json"):
            if (u / f).is_file():
                shutil.copy(u / f, meta / f)
        print(f"merged {u.name}")
    # profile-level files for generate_report.py come from the first unit (identical by construction)
    for f in ("config.yaml", "environment.json", "dataset_audit.json", "split_manifest.json"):
        if (units[0] / f).is_file() and not (a.out / f).exists():
            shutil.copy(units[0] / f, a.out / f)
    print(f"done -> {a.out}  (next: python scripts/generate_report.py --results {a.out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
