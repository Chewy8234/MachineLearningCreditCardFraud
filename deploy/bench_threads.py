"""Thread-count benchmark + numerical-reproducibility check for Bank B (RF) and Bank A (XGBoost).

    python deploy/bench_threads.py --config configs/paper.yaml --seed 42 --scenario moderate_non_iid \
        --threads 1 4 8 12 16 --out bench_threads.json

Run it on an otherwise idle machine (it measures wall-clock time). For every thread count it
(1) trains the local model and one KD-style model (local rows + soft-target proxy rows, the
exact expansion used in the federation), (2) records fit time and peak RSS, and (3) compares
validation probabilities with the reference thread count (the first value of --threads).
Also checks whether PREDICTING with different n_jobs changes one fixed model's output.
Read-only with respect to the project: it only reads the cached splits and writes --out.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heterofl.data.pipeline import prepare_bundle, prepare_scenario  # noqa: E402
from heterofl.distillation.soft_targets import build_distillation_set  # noqa: E402
from heterofl.models.model_factory import create_model, model_seed  # noqa: E402
from heterofl.utils.config import load_config  # noqa: E402


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform == "darwin" else r / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="moderate_non_iid")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 12])
    ap.add_argument("--out", type=Path, default=Path("bench_threads.json"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    data = prepare_scenario(cfg, prepare_bundle(cfg), a.seed, a.scenario)
    Xv, yv = data.validation()
    Xp, _ = data.proxy_features()
    kd = cfg["kd"]
    results = []
    ref: dict[tuple[str, str], np.ndarray] = {}
    for bank, mtype in (("A", "xgboost"), ("B", "random_forest")):
        X, y = data.bank(bank)
        seed = model_seed(a.seed, f"bank_{bank}", mtype)
        # a fixed pseudo-teacher (the local model's own proxy predictions) -- timing/determinism only
        base = create_model(mtype, cfg["models"], seed, n_jobs=a.threads[0])
        base.fit(X, y)
        teacher = base.predict_proba(Xp)
        for n in a.threads:
            row = {"bank": bank, "model": mtype, "n_jobs": n, "n_local": int(len(y))}
            for phase in ("local", "kd"):
                m = create_model(mtype, cfg["models"], seed, n_jobs=n)
                t0 = time.perf_counter()
                if phase == "local":
                    m.fit(X, y)
                else:
                    ds = build_distillation_set(X, y, Xp, teacher, kd["lambda_kd"], kd["prob_clip"])
                    m.fit(ds.X, ds.y, sample_weight=ds.sample_weight)
                row[f"{phase}_fit_s"] = time.perf_counter() - t0
                p = m.predict_proba(Xv)
                key = (bank, phase)
                ref.setdefault(key, p)
                row[f"{phase}_max_abs_dp_vs_ref"] = float(np.max(np.abs(p - ref[key])))
                row[f"{phase}_bitwise_equal_to_ref"] = bool(np.array_equal(p, ref[key]))
            row["peak_rss_mb_so_far"] = peak_rss_mb()
            print(json.dumps(row), flush=True)
            results.append(row)
        # predict-time thread effect on ONE fixed model
        if mtype == "random_forest":
            outs = []
            for n in a.threads:
                base.clf.n_jobs = n
                outs.append(base.predict_proba(Xv))
            results.append({"bank": bank, "model": mtype, "predict_only_max_abs_dp_across_threads":
                            float(max(np.max(np.abs(o - outs[0])) for o in outs))})
    a.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
