"""Merge retrieved NRP unit directories into one results tree the report generator understands.

    python deploy/merge_units.py results/nrp/nrp-paper-cpu12-v1 --out results/full_nrp_cpu12

Refuses to overwrite: a (seed, scenario) already present in --out aborts the merge (no duplicates).
Refuses to mix experiments: every unit must carry the same experiment identity (experiment.json),
and so must an existing --out. Per-unit hardware.json / ledger.jsonl / runner.log / config.yaml /
environment.json / experiment.json are kept under <out>/_units/<unit>/ so every result stays
traceable to its pod and CPU setting.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def experiment_hash(d: Path) -> str | None:
    f = d / "experiment.json"
    return json.loads(f.read_text()).get("experiment_hash") if f.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    units = sorted(a.tag_dir.glob("unit_*"))
    if not units:
        print(f"No unit_* directories in {a.tag_dir}", file=sys.stderr)
        return 2
    hashes = {u.name: experiment_hash(u) for u in units}
    if None in hashes.values():
        print(f"Units without experiment.json (legacy, unverifiable): "
              f"{[k for k, v in hashes.items() if v is None]}; aborting", file=sys.stderr)
        return 1
    if len(set(hashes.values())) != 1:
        print(f"Units belong to different experiments {hashes}; aborting", file=sys.stderr)
        return 1
    existing = experiment_hash(a.out)
    if a.out.exists() and any(a.out.iterdir()) and existing != next(iter(hashes.values())):
        print(f"{a.out} holds experiment {existing}, units are {next(iter(hashes.values()))}; "
              "aborting", file=sys.stderr)
        return 1
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
                  "dataset_audit.json", "split_manifest.json", "experiment.json"):
            if (u / f).is_file():
                shutil.copy(u / f, meta / f)
        print(f"merged {u.name}")
    # profile-level files for generate_report.py come from the first unit (identical by construction)
    for f in ("experiment.json", "config.yaml", "environment.json", "dataset_audit.json",
              "split_manifest.json"):
        if (units[0] / f).is_file() and not (a.out / f).exists():
            shutil.copy(units[0] / f, a.out / f)
    print(f"done -> {a.out}  (next: python scripts/generate_report.py --results {a.out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
