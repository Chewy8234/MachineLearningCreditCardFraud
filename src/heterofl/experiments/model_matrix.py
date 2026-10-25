"""Model-family matrix: every model family under every training regime, per round.

Complements the main runner (which assigns ONE model family per bank) with a
grid in which each of four families -- logistic regression, MLP, Random Forest,
XGBoost -- is trained

1. *local only* at every bank (metrics averaged over the banks),
2. *centralized* on the pooled client partitions (hypothetical reference),
3. *homogeneous FL* for that family, tracked for ``num_rounds`` rounds:
   * logistic regression / MLP -- FedAvg (example-weighted parameter mean),
   * Random Forest -- federated forest: every round each bank grows
     ``rf_trees_per_round`` new trees on its own data; the global forest is the
     union of all trees so far (equal-weight tree vote),
   * XGBoost -- federated bagging (Flower's ``FedXgbBagging`` scheme): every
     round each bank boosts ``xgb_trees_per_round`` trees starting from the
     current global model on its own data; the server appends all new trees,

and the proposed heterogeneous committee (bank model families as configured,
leave-one-out KD on the proxy set) is run for the same number of rounds.

Everything runs in-process, without Ray/Flower: the per-round computations are
the same functions the Flower apps call (``initial_fit``, ``distill_update``,
``compute_teachers``), only the message transport is skipped. Data access is
identical: banks see only their partition (+ proxy features), thresholds and
model selection use validation only (final round, fixed a priori), and test
data is read solely to report metrics. Per-round test AP is a reporting curve,
never used to pick a round.

Logistic regression is the project's ``MLPModel`` with no hidden layer (a single
linear unit + sigmoid, Adam, BCE), so FedAvg applies to it unchanged.

Usage::

    python -m heterofl.experiments.model_matrix --config configs/paper.yaml \\
        --seeds 42 --scenarios iid --out results/model_matrix
    python -m heterofl.experiments.model_matrix --config configs/paper.yaml \\
        --out results/model_matrix --report-only
"""

from __future__ import annotations

import heterofl  # noqa: F401  (import before torch/xgboost; see docs/architecture.md)

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from heterofl.data.pipeline import ScenarioData, prepare_bundle, prepare_scenario
from heterofl.distillation.aggregation import (
    compute_teachers,
    validation_ap_weights,
    weighted_average,
)
from heterofl.distillation.trainers import distill_update, initial_fit
from heterofl.evaluation.metrics import average_precision, evaluate_system
from heterofl.federation.messages import PREDICTION_DTYPE
from heterofl.federation.xgb_bagging import FedXgbBagging, logit, xgb_native_params
from heterofl.models.mlp import MLPModel
from heterofl.models.model_factory import create_model, model_seed
from heterofl.models.random_forest import RandomForestModel
from heterofl.utils.config import load_config, resolve_path
from heterofl.utils.io import write_csv, write_json
from heterofl.utils.logging import get_logger, setup_logging
from heterofl.utils.reproducibility import derive_seed, git_info

log = get_logger("experiments.model_matrix")

MODELS = ("logistic_regression", "mlp", "random_forest", "xgboost")
LABELS = {"logistic_regression": "Logistic Regression", "mlp": "MLP",
          "random_forest": "Random Forest", "xgboost": "XGBoost"}
FL_LABELS = {"logistic_regression": "Logistic Regression (FedAvg)", "mlp": "MLP (FedAvg)",
             "random_forest": "Random Forest (federated forest)",
             "xgboost": "XGBoost (federated bagging)"}
SHORT = {"xgboost": "XGBoost", "random_forest": "RF", "mlp": "MLP",
         "logistic_regression": "LR"}
EXPERIMENTS = ("local", "centralized", "federated", "committee")

#: Matrix-specific settings (everything else comes from the profile config).
SETTINGS = {
    "num_rounds": 100,
    "report_rounds": [1, 5, 10, 25, 50, 75, 100],
    "fedavg_local_epochs": 1,     # per round, LR and MLP
    "rf_trees_per_round": 1,      # per bank per round -> 3 trees/round, 300 after 100
    "xgb_trees_per_round": 1,     # per bank per round -> 3 trees/round, 300 after 100
    "logistic_regression": {"hidden_sizes": [], "dropout": 0.0, "learning_rate": 0.01,
                            "weight_decay": 0.0, "batch_size": 512, "epochs": 10,
                            "device": "cpu"},
}


# --------------------------------------------------------------------------- helpers
class Harness:
    """Evaluates probability vectors: validation-chosen max-F1 threshold, test metrics."""

    def __init__(self, cfg: dict, data: ScenarioData):
        self.cfg = cfg
        self.data = data
        self.Xv, self.yv = data.validation()
        self.Xt, self.yt = data.test()
        self.rule = cfg["evaluation"].get("threshold_rule", "max_f1")

    def final(self, p_val, p_test) -> dict:
        ev = evaluate_system(self.yv, p_val, self.yt, p_test, rule=self.rule, fpr_targets=())
        t = ev["test"]
        m = t["at_val_threshold"]
        return {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "average_precision": t["average_precision"], "roc_auc": t["roc_auc"],
                "val_average_precision": ev["validation"]["average_precision"],
                "threshold": ev["validation_threshold"]}

    def curve(self, p_val, p_test) -> dict:
        return {"val_average_precision": average_precision(self.yv, p_val),
                "test_average_precision": average_precision(self.yt, p_test)}


def _lr_or_mlp(mtype: str, cfg: dict, seed: int, n_features: int) -> MLPModel:
    params = SETTINGS["logistic_regression"] if mtype == "logistic_regression" \
        else cfg["models"]["mlp"]
    return MLPModel(params, seed, n_features=n_features)


def make_model(mtype: str, cfg: dict, seed: int, n_jobs: int, n_features: int):
    if mtype in ("logistic_regression", "mlp"):
        return _lr_or_mlp(mtype, cfg, seed, n_features)
    return create_model(mtype, cfg["models"], seed, n_jobs=n_jobs, n_features=n_features)


_logit = logit


def _wire(p: np.ndarray) -> np.ndarray:
    """Round-trip through the Flower message dtype, as the real KD transport does."""
    return np.asarray(p, dtype=PREDICTION_DTYPE).astype(np.float64)


def _report_round(r: int, num_rounds: int) -> bool:
    return r in SETTINGS["report_rounds"] or r == num_rounds or r <= 10 or r % 5 == 0


# --------------------------------------------------------------------------- experiments
def run_local(cfg, data, seed, n_jobs, h: Harness) -> list[dict]:
    rows = []
    for bank in data.banks:
        X, y = data.bank(bank)
        for mtype in MODELS:
            t0 = time.perf_counter()
            model = make_model(mtype, cfg, model_seed(seed, f"bank_{bank}", mtype), n_jobs,
                               X.shape[1])
            initial_fit(model, X, y)
            res = h.final(model.predict_proba(h.Xv), model.predict_proba(h.Xt))
            rows.append({"experiment": "local", "model": mtype, "bank": bank,
                         "train_seconds": time.perf_counter() - t0, **res})
            log.info("local   bank %s %-20s AP=%.4f F1=%.4f", bank, mtype,
                     res["average_precision"], res["f1"])
    return rows


def run_centralized(cfg, data, seed, n_jobs, h: Harness) -> list[dict]:
    X, y = data.pooled_clients()
    rows = []
    for mtype in MODELS:
        t0 = time.perf_counter()
        model = make_model(mtype, cfg, model_seed(seed, "centralized", mtype), n_jobs, X.shape[1])
        initial_fit(model, X, y)
        res = h.final(model.predict_proba(h.Xv), model.predict_proba(h.Xt))
        rows.append({"experiment": "centralized", "model": mtype, "bank": "pooled",
                     "train_seconds": time.perf_counter() - t0, **res})
        log.info("central %-20s AP=%.4f F1=%.4f", mtype, res["average_precision"], res["f1"])
    return rows


def _fedavg(mtype, cfg, data, seed, h, num_rounds):
    """FedAvg: example-weighted mean of the banks' parameters after local epochs."""
    banks = {b: data.bank(b) for b in data.banks}
    n_features = h.Xv.shape[1]
    glob = _lr_or_mlp(mtype, cfg, model_seed(seed, "fedavg_global", mtype), n_features)
    n = np.array([len(banks[b][1]) for b in data.banks], dtype=np.float64)
    w = n / n.sum()
    for r in range(1, num_rounds + 1):
        states = []
        for b in data.banks:
            client = _lr_or_mlp(mtype, cfg, derive_seed(seed, "fedavg-client", mtype, b, r),
                                n_features)
            client.set_state_dict(glob.get_state_dict())
            client.fit(*banks[b], epochs=int(SETTINGS["fedavg_local_epochs"]))
            states.append(client.get_state_dict())
        glob.set_state_dict({k: sum(wk * s[k] for wk, s in zip(w, states, strict=True))
                             for k in states[0]})
        yield r, glob.predict_proba(h.Xv), glob.predict_proba(h.Xt)


def _federated_forest(cfg, data, seed, n_jobs, h, num_rounds):
    """Union forest: each bank adds ``rf_trees_per_round`` trees per round."""
    k = int(SETTINGS["rf_trees_per_round"])
    params = {**cfg["models"]["random_forest"], "n_estimators": k}
    banks = {b: data.bank(b) for b in data.banks}
    sum_v, sum_t, n_trees = np.zeros(len(h.yv)), np.zeros(len(h.yt)), 0
    for r in range(1, num_rounds + 1):
        for b in data.banks:
            rf = RandomForestModel(params, derive_seed(seed, "federated-forest", b, r), n_jobs)
            rf.fit(*banks[b])
            sum_v += k * rf.predict_proba(h.Xv)
            sum_t += k * rf.predict_proba(h.Xt)
            n_trees += k
        yield r, sum_v / n_trees, sum_t / n_trees


def _xgb_native_params(cfg, seed, n_jobs) -> dict:
    return xgb_native_params(cfg["models"]["xgboost"], seed, n_jobs)


def _federated_xgb_bagging(cfg, data, seed, n_jobs, h, num_rounds):
    """Federated bagging (``heterofl.federation.xgb_bagging``): new trees are appended.

    The per-round test curve is a reporting curve only (see the module docstring).
    """
    fx = FedXgbBagging({b: data.bank(b) for b in data.banks}, cfg["models"]["xgboost"],
                       seed, n_jobs, num_rounds, int(SETTINGS["xgb_trees_per_round"]))
    for r, p, _ in fx.rounds({"val": h.Xv, "test": h.Xt}):
        yield r, p["val"], p["test"]


def run_federated(cfg, data, seed, n_jobs, h: Harness, num_rounds: int):
    rows, curve = [], []
    runs = {
        "logistic_regression": lambda: _fedavg("logistic_regression", cfg, data, seed, h,
                                               num_rounds),
        "mlp": lambda: _fedavg("mlp", cfg, data, seed, h, num_rounds),
        "random_forest": lambda: _federated_forest(cfg, data, seed, n_jobs, h, num_rounds),
        "xgboost": lambda: _federated_xgb_bagging(cfg, data, seed, n_jobs, h, num_rounds),
    }
    for mtype, gen in runs.items():
        t0 = time.perf_counter()
        for r, p_val, p_test in gen():
            if _report_round(r, num_rounds):
                curve.append({"experiment": "federated", "model": mtype, "bank": "global",
                              "round": r, **h.curve(p_val, p_test)})
            if r == num_rounds:
                res = h.final(p_val, p_test)
                rows.append({"experiment": "federated", "model": mtype, "bank": "global",
                             "round": r, "train_seconds": time.perf_counter() - t0, **res})
                log.info("FL      %-20s round %d AP=%.4f F1=%.4f (%.0fs)", mtype, r,
                         res["average_precision"], res["f1"], time.perf_counter() - t0)
    return rows, curve


def run_committee(cfg, data, seed, n_jobs, h: Harness, num_rounds: int):
    """Heterogeneous leave-one-out KD on the proxy set (in-process replica of the Flower run)."""
    kd = cfg["kd"]
    clients = {c["bank"]: c["model"] for c in cfg["clients"]}
    Xp, _ = data.proxy_features()
    banks = {b: data.bank(b) for b in clients}
    models = {}
    for b, mtype in clients.items():
        models[b] = create_model(mtype, cfg["models"], model_seed(seed, f"bank_{b}", mtype),
                                 n_jobs=n_jobs, n_features=Xp.shape[1])
        initial_fit(models[b], *banks[b])

    rows, curve = [], []
    t0 = time.perf_counter()
    for r in range(0, num_rounds + 1):
        if r > 0:
            weights = None
            if kd["teacher_weighting"] == "validation_ap":
                weights = validation_ap_weights({b: average_precision(h.yv, pv[b])
                                                 for b in clients})
            teachers = compute_teachers(proxy, kd["teacher"], len(Xp), weights)
            for b in clients:
                distill_update(models[b], *banks[b], Xp, _wire(teachers[b]), kd)
        proxy = {b: _wire(m.predict_proba(Xp)) for b, m in models.items()}
        pv = {b: m.predict_proba(h.Xv) for b, m in models.items()}
        pt = {b: m.predict_proba(h.Xt) for b, m in models.items()}
        ens_w = None
        if cfg["ensemble"].get("weighting", "equal") == "validation_ap":
            ens_w = validation_ap_weights({b: average_precision(h.yv, pv[b]) for b in clients})
        members = sorted(clients)
        ens_v = weighted_average(pv, members, ens_w)
        ens_t = weighted_average(pt, members, ens_w)
        if _report_round(r, num_rounds) or r == 0:
            points = {b: h.curve(pv[b], pt[b]) for b in members}
            points["committee"] = h.curve(ens_v, ens_t)
            for b, c in points.items():
                curve.append({"experiment": "committee", "bank": b, "round": r,
                              "model": clients.get(b, "ensemble"), **c})
            log.info("KD      round %3d %s (%.0fs)", r,
                     " ".join(f"{b}:{c['test_average_precision']:.4f}" for b, c in points.items()),
                     time.perf_counter() - t0)
        if r in (0, num_rounds):
            phase = "round0" if r == 0 else "final"
            for b in members:
                rows.append({"experiment": "committee", "model": clients[b], "bank": b,
                             "round": r, "phase": phase, **h.final(pv[b], pt[b])})
            rows.append({"experiment": "committee", "model": "ensemble", "bank": "committee",
                         "round": r, "phase": phase, **h.final(ens_v, ens_t)})
    return rows, curve


# --------------------------------------------------------------------------- driver
def run_unit(cfg: dict, seed: int, scenario: str, out: Path, experiments, n_jobs: int,
             num_rounds: int, overwrite: bool) -> None:
    bundle = prepare_bundle(cfg)
    data = prepare_scenario(cfg, bundle, seed, scenario)
    h = Harness(cfg, data)
    d = out / f"seed_{seed}" / scenario
    for exp in experiments:
        final_path, curve_path = d / f"{exp}.csv", d / f"{exp}_rounds.csv"
        if final_path.is_file() and not overwrite:
            log.info("seed %d %s: %s already done (%s); skipping", seed, scenario, exp, final_path)
            continue
        log.info("=== seed %d | %s | %s ===", seed, scenario, exp)
        t0 = time.perf_counter()
        if exp == "local":
            rows, curve = run_local(cfg, data, seed, n_jobs, h), []
        elif exp == "centralized":
            rows, curve = run_centralized(cfg, data, seed, n_jobs, h), []
        elif exp == "federated":
            rows, curve = run_federated(cfg, data, seed, n_jobs, h, num_rounds)
        else:
            rows, curve = run_committee(cfg, data, seed, n_jobs, h, num_rounds)
        meta = {"seed": seed, "scenario": scenario}
        if curve:
            write_csv(curve_path, [{**meta, **c} for c in curve])
        write_csv(final_path, [{**meta, **r} for r in rows])  # written last = completion marker
        log.info("seed %d %s: %s finished in %.0fs", seed, scenario, exp, time.perf_counter() - t0)


# --------------------------------------------------------------------------- report
def _load(out: Path, pattern: str) -> pd.DataFrame:
    files = sorted(out.glob(f"seed_*/*/{pattern}"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _fmt(v) -> str:
    return "  -  " if pd.isna(v) else f"{v:.3f}"


def _main_table(final: pd.DataFrame) -> list[str]:
    cols = ("precision", "recall", "f1", "average_precision")
    head = (f"{'Experiment':<30} {'Federated?':<11} {'Model':<36} "
            f"{'Precision':<10} {'Recall':<7} {'F1':<6} {'PR-AUC':<6}")
    sep = "─" * len(head)
    lines = [head]

    def block(title, fed, entries):
        if not entries:
            return
        if len(lines) > 1:
            lines.append(sep)
        for i, (label, sub) in enumerate(entries):
            # mean over banks within each seed, then over seeds
            m = sub.groupby("seed")[list(cols)].mean().mean()
            lines.append(f"{title if i == 0 else '':<30} {fed if i == 0 else '':<11} {label:<36} "
                         f"{_fmt(m['precision']):<10} {_fmt(m['recall']):<7} "
                         f"{_fmt(m['f1']):<6} {_fmt(m['average_precision']):<6}")

    def entries(exp, labels, extra=None):
        out = []
        for mtype in MODELS:
            sub = final[(final.experiment == exp) & (final.model == mtype)]
            if extra is not None:
                sub = sub[extra(sub)]
            if len(sub):
                out.append((labels[mtype], sub))
        return out

    block("1. Local only", "No", entries("local", LABELS))
    block("2. Centralized", "No", entries("centralized", LABELS))
    block("3. Homogeneous FL", "Yes", entries("federated", FL_LABELS))
    kd = final[(final.experiment == "committee") & (final.phase == "final")]
    if len(kd):
        members = kd[kd.bank != "committee"]
        names = " + ".join(SHORT[m] for m in
                           members.sort_values("bank").drop_duplicates("bank").model)
        block("4. Committee (KD)", "Yes",
              [(f"{names} (bank mean)", members),
               ("Committee ensemble (avg. of the 3)", kd[kd.bank == "committee"])])
    return lines


def _per_bank_table(final: pd.DataFrame) -> list[str]:
    kd = final[final.experiment == "committee"]
    kd = kd[kd.bank != "committee"]
    if not len(kd):
        return []
    lines = [f"{'Bank':<6}{'Model':<15}{'Local only':<12}{'Committee':<11}{'Change':<7}"]
    for bank, sub in kd.groupby("bank"):
        before = sub[sub.phase == "round0"].average_precision.mean()
        after = sub[sub.phase == "final"].average_precision.mean()
        lines.append(f"{bank:<6}{LABELS[sub.model.iloc[0]]:<15}{_fmt(before):<12}"
                     f"{_fmt(after):<11}{after - before:+.3f}")
    return lines


def _rounds_table(curve: pd.DataFrame, rounds: list[int]) -> list[str]:
    if not len(curve):
        return []
    rounds = [r for r in rounds if r in set(curve["round"])]
    head = f"{'Model':<23}" + "".join(f"{'Round ' + str(r):<10}" for r in rounds)
    lines = [head]
    fl = curve[curve.experiment == "federated"]
    for mtype in ("logistic_regression", "mlp", "xgboost", "random_forest"):
        sub = fl[fl.model == mtype]
        if len(sub):
            m = sub.groupby("round").test_average_precision.mean()
            lines.append(f"{LABELS[mtype]:<23}" + "".join(f"{_fmt(m.get(r)):<10}" for r in rounds))
    kd = curve[curve.experiment == "committee"]
    for label, sub in (("Committee (bank mean)", kd[kd.bank != "committee"]),
                       ("Committee ensemble", kd[kd.bank == "committee"])):
        if len(sub):
            m = sub.groupby(["seed", "round"]).test_average_precision.mean().groupby("round").mean()
            lines.append(f"{label:<23}" + "".join(f"{_fmt(m.get(r)):<10}" for r in rounds))
    return lines


def write_report(out: Path, cfg: dict) -> Path:
    final = pd.concat([_load(out, f"{e}.csv") for e in EXPERIMENTS], ignore_index=True)
    curve = _load(out, "*_rounds.csv")
    if not len(final):
        raise FileNotFoundError(f"No matrix results under {out}")
    if "phase" not in final:
        final["phase"] = np.nan
    git = git_info(cfg["project_root"])
    parts = [f"# Model-family matrix: `{out.name}`", "",
             f"Git commit `{git.get('commit')}` (dirty: {git.get('dirty')}). "
             f"Rounds: {SETTINGS['num_rounds']}. Threshold: max-F1 on validation, applied to "
             "test. PR-AUC = average precision (test). Model selection: final round.", ""]
    for scenario, fs in final.groupby("scenario", sort=False):
        cs = curve[curve.scenario == scenario] if len(curve) else curve
        seeds = sorted(fs.seed.unique())
        parts += [f"## Scenario `{scenario}` (seeds {', '.join(map(str, seeds))}; "
                  "mean over seeds)", "",
                  "**Results (test set, mean across the 3 banks)**", "", "```text",
                  *_main_table(fs), "```", ""]
        pb = _per_bank_table(fs)
        if pb:
            parts += ["**Committee per bank (each bank keeps its own model type; test PR-AUC)**",
                      "", "```text", *pb, "```", ""]
        rt = _rounds_table(cs, SETTINGS["report_rounds"])
        if rt:
            parts += ["**Test PR-AUC by round (federated experiments)**", "", "```text", *rt,
                      "```", ""]
    path = out / "tables.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    final.to_csv(out / "all_final.csv", index=False)
    if len(curve):
        curve.to_csv(out / "all_rounds.csv", index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Model-family x training-regime matrix")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", default="results/model_matrix")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--scenarios", nargs="+")
    p.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    p.add_argument("--rounds", type=int, default=SETTINGS["num_rounds"])
    p.add_argument("--n-jobs", type=int, default=None, help="threads for RF/XGBoost")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config, args.set)
    SETTINGS["num_rounds"] = args.rounds
    out = resolve_path(cfg, args.out)
    if not args.report_only:
        seeds = args.seeds or list(cfg["seeds"])
        scenarios = args.scenarios or [s for s, v in cfg["scenarios"].items()
                                       if not v.get("ablation")]
        n_jobs = args.n_jobs or int(cfg["federation"]["client_num_cpus"])
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "settings.json", {**SETTINGS, "config": cfg["config_path"],
                                           "overrides": args.set})
        for seed in seeds:
            for scenario in scenarios:
                run_unit(cfg, seed, scenario, out, args.experiments, n_jobs, args.rounds,
                         args.overwrite)
    path = write_report(out, cfg)
    print(path.read_text(encoding="utf-8"))
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
