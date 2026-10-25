"""Heterogeneous vs homogeneous committees: one paired comparison across results directories.

    python scripts/compare_committees.py --hetero results/full_v2 \\
        --homog-xgboost results/homog_xgboost --homog-random-forest results/homog_random_forest \\
        --homog-mlp results/homog_mlp --out results/committee_comparison

Every profile (``configs/paper.yaml``, ``configs/homog_*.yaml``) is its own results
directory with its own experiment identity. Before any number is compared, this module
verifies (``ComparisonError`` otherwise) that all directories

* carry an experiment identity, the same dataset SHA-256, the same split bundle and the
  same library versions,
* differ in their semantic configuration only in ``clients`` / ``profile`` / ``fedxgb``,
* hold, for every (seed, scenario), runs on the byte-identical partition and the same
  test labels.

Only completed runs contribute (``report.collect``). Differences are seed-paired;
the paired bootstrap resamples the shared test set per seed. Both are descriptive:
three seeds share one test set, so no significance tests are reported.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from heterofl.evaluation import visualization as viz
from heterofl.evaluation.latex_tables import bold_best, escape, fmt_mean_std, latex_table, write_table
from heterofl.evaluation.report import collect, order_scenarios, scenario_labels
from heterofl.evaluation.statistics import paired_bootstrap_ap_diff
from heterofl.utils.config import config_diff
from heterofl.utils.io import load_npz, read_json, write_csv, write_json
from heterofl.utils.logging import get_logger

log = get_logger("report.committees")

ROLES = ("hetero", "xgboost", "random_forest", "mlp")
ROLE_LABELS = {"hetero": "Heterogeneous (XGB + RF + MLP)", "xgboost": "Homogeneous 3xXGBoost",
               "random_forest": "Homogeneous 3xRandom Forest", "mlp": "Homogeneous 3xMLP"}
#: semantic configuration keys allowed to differ between the compared profiles
ALLOWED_CONFIG_DIFF = {"clients", "profile", "fedxgb"}
TOP_K = 200   # diversity: overlap of the members' top-k most suspicious validation rows


class ComparisonError(RuntimeError):
    """The results directories cannot be compared on equal terms."""


@dataclass(frozen=True)
class System:
    key: str
    label: str
    group: str
    role: str
    variant: str
    system: str
    family: str       # colour: architecture of the members ("hetero" or a model type)
    kind: str         # marker: "kd" | "ensemble" | "fl" | "control"


SYSTEMS = [
    System("fedavg", "FedAvg MLP (5x2 epochs)", "Homogeneous FL", "hetero", "primary",
           "fedavg_global", "mlp", "fl"),
    System("fedavg_20ep", "FedAvg MLP (5x4 epochs, epoch-matched)", "Homogeneous FL", "mlp",
           "fedavg_20_epochs", "fedavg_global", "mlp", "fl"),
    System("fedxgb", "Federated XGBoost (tree bagging)", "Homogeneous FL", "xgboost", "primary",
           "fedxgb_global", "xgboost", "fl"),
    System("kd_xgb", "Homogeneous XGB KD committee", "Homogeneous KD committee", "xgboost",
           "primary", "kd_committee", "xgboost", "kd"),
    System("kd_rf", "Homogeneous RF KD committee", "Homogeneous KD committee", "random_forest",
           "primary", "kd_committee", "random_forest", "kd"),
    System("kd_mlp", "Homogeneous MLP KD committee", "Homogeneous KD committee", "mlp",
           "primary", "kd_committee", "mlp", "kd"),
    System("ens_xgb", "Homogeneous XGB ensemble (no KD)", "Homogeneous ensemble (no KD)",
           "xgboost", "primary", "ensemble_equal", "xgboost", "ensemble"),
    System("ens_rf", "Homogeneous RF ensemble (no KD)", "Homogeneous ensemble (no KD)",
           "random_forest", "primary", "ensemble_equal", "random_forest", "ensemble"),
    System("ens_mlp", "Homogeneous MLP ensemble (no KD)", "Homogeneous ensemble (no KD)", "mlp",
           "primary", "ensemble_equal", "mlp", "ensemble"),
    System("ens_het", "Heterogeneous ensemble (no KD)", "Heterogeneous", "hetero", "primary",
           "ensemble_equal", "hetero", "ensemble"),
    System("kd_het", "Heterogeneous KD committee (proposed)", "Heterogeneous", "hetero",
           "primary", "kd_committee", "hetero", "kd"),
    System("kd_het_l0", "Heterogeneous KD, lambda=0 (control)", "Controls", "hetero",
           "kd_lambda_0.0", "kd_committee", "hetero", "control"),
    System("kd_mlp_l0", "Homogeneous MLP KD, lambda=0 (control)", "Controls", "mlp",
           "kd_lambda_0.0", "kd_committee", "mlp", "control"),
]
REFERENCE = [
    System("central_xgb", "Centralized XGBoost (ref.)", "Reference", "hetero", "primary",
           "centralized_xgboost", "xgboost", "ref"),
    System("central_rf", "Centralized Random Forest (ref.)", "Reference", "hetero", "primary",
           "centralized_random_forest", "random_forest", "ref"),
    System("central_mlp", "Centralized MLP (ref.)", "Reference", "hetero", "primary",
           "centralized_mlp", "mlp", "ref"),
]
PROPOSED = "kd_het"
#: (question, a, b): seed-paired AP(a) - AP(b)
COMPARISONS = [
    ("Q1 proposed vs homogeneous FL", PROPOSED, "fedavg"),
    ("Q1 proposed vs homogeneous FL", PROPOSED, "fedavg_20ep"),
    ("Q1 proposed vs homogeneous FL", PROPOSED, "fedxgb"),
    ("Q2 proposed vs homogeneous KD", PROPOSED, "kd_xgb"),
    ("Q2 proposed vs homogeneous KD", PROPOSED, "kd_rf"),
    ("Q2 proposed vs homogeneous KD", PROPOSED, "kd_mlp"),
    ("Q2b proposed vs homogeneous ensembles", PROPOSED, "ens_xgb"),
    ("Q2b proposed vs homogeneous ensembles", PROPOSED, "ens_rf"),
    ("Q2b proposed vs homogeneous ensembles", PROPOSED, "ens_mlp"),
    ("Q3 heterogeneity without KD", "ens_het", "ens_xgb"),
    ("Q3 heterogeneity without KD", "ens_het", "ens_rf"),
    ("Q3 heterogeneity without KD", "ens_het", "ens_mlp"),
    ("Q4 KD vs matched no-KD committee", PROPOSED, "ens_het"),
    ("Q4 KD vs matched no-KD committee", "kd_xgb", "ens_xgb"),
    ("Q4 KD vs matched no-KD committee", "kd_rf", "ens_rf"),
    ("Q4 KD vs matched no-KD committee", "kd_mlp", "ens_mlp"),
    ("Q4 KD vs matched no-KD committee", PROPOSED, "kd_het_l0"),
    ("Q4 KD vs matched no-KD committee", "kd_mlp", "kd_mlp_l0"),
]
SECONDARY = [("test_roc_auc", "ROC-AUC"), ("test_at_val_threshold_precision", "Prec."),
             ("test_at_val_threshold_recall", "Rec."), ("test_at_val_threshold_f1", "F1"),
             ("test_at_val_threshold_fp", "FP"), ("test_at_val_threshold_fn", "FN")]
CONFUSION = ["test_at_val_threshold_tp", "test_at_val_threshold_fp",
             "test_at_val_threshold_tn", "test_at_val_threshold_fn"]
FAMILY_COLORS = {"hetero": "#2a78d6", "xgboost": "#eb6834", "random_forest": "#1baf7a",
                 "mlp": "#eda100"}   # reference categorical slots 1-4 (CVD-validated order)
FAMILY_NAMES = {"hetero": "Heterogeneous", "xgboost": "XGBoost", "random_forest": "Random Forest",
                "mlp": "MLP"}
KIND_MARKERS = {"kd": "o", "ensemble": "s", "fl": "^", "control": "D", "ref": "x"}
KIND_NAMES = {"kd": "KD committee", "ensemble": "Ensemble (no KD)", "fl": "Homogeneous FL",
              "control": "λ=0 control", "ref": "Centralized (ref.)"}


# --------------------------------------------------------------------------- loading
def load_roles(dirs: dict[str, Path]) -> dict[str, dict]:
    out = {}
    for role, d in dirs.items():
        d = Path(d).resolve()
        exp = d / "experiment.json"
        if not exp.is_file():
            raise ComparisonError(f"{d} has no experiment.json (legacy or not a results "
                                  "directory); only identity-recorded results are compared")
        data = collect(d)
        if data["metrics"].empty:
            raise ComparisonError(f"{d}: no completed runs")
        out[role] = {"dir": d, "experiment": read_json(exp), "split": read_json(d / "split_manifest.json"),
                     **data}
    return out


def verify(roles: dict[str, dict]) -> dict:
    """Raise unless every directory compares on equal terms; return the check record."""
    names = list(roles)
    ref = roles["hetero" if "hetero" in roles else names[0]]
    rid = ref["experiment"]["identity"]
    checks = {"directories": {r: str(v["dir"]) for r, v in roles.items()}}
    for r, v in roles.items():
        ident = v["experiment"]["identity"]
        if v["experiment"]["dataset_sha256"] != ref["experiment"]["dataset_sha256"]:
            raise ComparisonError(f"{r}: dataset SHA-256 differs")
        if v["split"]["tag"] != ref["split"]["tag"] or v["split"]["splits"] != ref["split"]["splits"]:
            raise ComparisonError(f"{r}: split bundle differs ({v['split']['tag']} vs "
                                  f"{ref['split']['tag']})")
        if ident["packages"] != rid["packages"]:
            raise ComparisonError(f"{r}: library versions differ: "
                                  f"{config_diff(rid['packages'], ident['packages'])}")
        diff = set(config_diff(rid["config"], ident["config"])) - ALLOWED_CONFIG_DIFF
        if diff:
            raise ComparisonError(f"{r}: configuration differs from {names[0]} beyond the "
                                  f"client model families: {sorted(diff)}")
        checks.setdefault("clients", {})[r] = [c["model"] for c in ident["config"]["clients"]]
    # identical realised partitions and test labels per (seed, scenario)
    allm = pd.concat([v["metrics"].assign(role=r) for r, v in roles.items()], ignore_index=True)
    parts = allm.groupby(["seed", "scenario"])["partition_sha256"].nunique()
    bad = parts[parts > 1]
    if len(bad):
        raise ComparisonError(f"Different partitions for the same seed/scenario: {list(bad.index)}")
    y_ref: dict = {}
    for _, r in allm.drop_duplicates(["role", "seed", "scenario"]).iterrows():
        f = Path(r["run_dir"]) / "predictions" / f"{r['system']}.npz"
        y = load_npz(f)["y_test"]
        key = (r["seed"], r["scenario"])
        if key in y_ref and not np.array_equal(y_ref[key], y):
            raise ComparisonError(f"Test labels differ at {key}")
        y_ref.setdefault(key, y)
    checks.update({"dataset_sha256": ref["experiment"]["dataset_sha256"],
                   "split_tag": ref["split"]["tag"], "packages": rid["packages"],
                   "partitions_identical": True, "test_labels_identical": True,
                   "n_seed_scenarios": int(len(parts))})
    return checks


def provenance(roles: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for r, v in roles.items():
        st = v["status"]
        done = st[st["status"] == "completed"]
        srcs = done[["source_commit", "source_dirty", "source_diff_sha256"]].drop_duplicates()
        rows.append({"role": r, "directory": str(v["dir"]),
                     "experiment_hash": v["experiment"]["experiment_hash"],
                     "runs_completed": len(done), "runs_not_completed": int(len(st) - len(done)),
                     "source": "; ".join(f"{(s.source_commit or '?')[:12]}"
                                         f"{' dirty ' + str(s.source_diff_sha256)[:12] if s.source_dirty else ''}"
                                         for s in srcs.itertuples()),
                     "all_from_clean_commit": bool(done["source_verified"].all()),
                     "total_run_wall_hours": float(done["wall_seconds"].sum()) / 3600})
    return pd.DataFrame(rows)


def system_rows(roles: dict[str, dict], specs: list[System]) -> pd.DataFrame:
    """One row per (system spec, seed, scenario), with the run's metrics."""
    out = []
    for s in specs:
        if s.role not in roles:
            continue
        m = roles[s.role]["metrics"]
        g = m[(m["variant"] == s.variant) & (m["system"] == s.system)
              & (m["kd_phase"].fillna("after") != "before")]
        if g.duplicated(["seed", "scenario"]).any():
            raise ComparisonError(f"{s.key}: several rows per seed/scenario")
        out.append(g.assign(key=s.key, label=s.label, group=s.group, role=s.role,
                            family_color=s.family, kind=s.kind))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- statistics
def seed_summary(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    recs = []
    for (key, sc), g in rows.groupby(["key", "scenario"], sort=False):
        rec = {"key": key, "label": g["label"].iloc[0], "group": g["group"].iloc[0],
               "scenario": sc, "n_seeds": g["seed"].nunique(),
               "seeds": ",".join(map(str, sorted(g["seed"])))}
        for m in metrics:
            v = g[m].astype(float)
            rec[f"{m}_mean"], rec[f"{m}_std"] = v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def paired(rows: pd.DataFrame, cfg_boot: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed-paired AP differences and per-seed paired bootstrap CIs for ``COMPARISONS``."""
    B = int(cfg_boot.get("n_resamples", 1000))
    conf = float(cfg_boot.get("confidence", 0.95))
    bseed = int(cfg_boot.get("seed", 0))
    summ, boot = [], []
    idx = rows.set_index(["key", "scenario", "seed"]).sort_index()
    for question, a, b in COMPARISONS:
        for sc in rows["scenario"].unique():
            try:
                ra, rb = idx.loc[(a, sc)], idx.loc[(b, sc)]
            except KeyError:
                continue
            seeds = sorted(set(ra.index) & set(rb.index))
            if not seeds:
                continue
            diffs, n_pos, n_neg = [], 0, 0
            for seed in seeds:
                xa, xb = ra.loc[seed], rb.loc[seed]
                if xa["partition_sha256"] != xb["partition_sha256"]:
                    raise ComparisonError(f"{a} vs {b} at {sc}/{seed}: different partitions")
                pa = load_npz(Path(xa["run_dir"]) / "predictions" / f"{xa['system']}.npz")
                pb = load_npz(Path(xb["run_dir"]) / "predictions" / f"{xb['system']}.npz")
                if not np.array_equal(pa["y_test"], pb["y_test"]):
                    raise ComparisonError(f"{a} vs {b} at {sc}/{seed}: test labels differ")
                d = float(xa["test_average_precision"] - xb["test_average_precision"])
                bs = paired_bootstrap_ap_diff(pa["y_test"], pa["p_test"], pb["p_test"], B, conf,
                                              bseed)
                if abs(bs["diff"] - d) > 1e-9:
                    raise ComparisonError(f"{a} vs {b}: stored AP does not match predictions")
                n_pos += bs["ci_low"] > 0
                n_neg += bs["ci_high"] < 0
                boot.append({"question": question, "a": a, "b": b, "scenario": sc, "seed": seed,
                             **bs})
                diffs.append(d)
            d = np.array(diffs)
            base = rb["test_average_precision"].loc[seeds].mean()
            summ.append({"question": question, "a": a, "b": b, "scenario": sc,
                         "n_pairs": len(d), "mean_diff": d.mean(),
                         "std_diff": d.std(ddof=1) if len(d) > 1 else np.nan,
                         "min_diff": d.min(), "max_diff": d.max(),
                         "relative_diff_pct": 100 * d.mean() / base,
                         "n_a_better": int((d > 0).sum()),
                         "n_seeds_ci_above_0": int(n_pos), "n_seeds_ci_below_0": int(n_neg)})
    return pd.DataFrame(summ), pd.DataFrame(boot)


def per_bank(roles: dict[str, dict]) -> pd.DataFrame:
    recs = []
    for role, v in roles.items():
        m = v["metrics"][v["metrics"]["variant"] == "primary"]
        for (sc, bank_sys), g in m[m["system"].str.match(r"bank_[A-Z]$")].groupby(
                ["scenario", "system"]):
            a = m[(m["scenario"] == sc) & (m["system"] == f"kd_{bank_sys}")]
            j = g.set_index("seed")["test_average_precision"].to_frame("before").join(
                a.set_index("seed")["test_average_precision"].to_frame("after"), how="inner")
            if j.empty:
                continue
            delta = j["after"] - j["before"]
            recs.append({"role": role, "scenario": sc, "bank": bank_sys[-1],
                         "model_type": g["model_type"].iloc[0], "n_seeds": len(j),
                         "before_mean": j["before"].mean(), "after_mean": j["after"].mean(),
                         "delta_mean": delta.mean(),
                         "delta_std": delta.std(ddof=1) if len(j) > 1 else np.nan,
                         "delta_min": delta.min(), "delta_max": delta.max(),
                         "n_improved": int((delta > 0).sum())})
    return pd.DataFrame(recs)


def _diversity(preds: list[np.ndarray]) -> tuple[float, float]:
    rho = np.mean([spearmanr(a, b).statistic for a, b in itertools.combinations(preds, 2)])
    tops = [set(np.argsort(-p)[:TOP_K]) for p in preds]
    jac = np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)])
    return float(rho), float(jac)


def diversity(roles: dict[str, dict]) -> pd.DataFrame:
    """Member agreement on VALIDATION predictions, before (local) and after KD."""
    recs = []
    for role, v in roles.items():
        for sdir in sorted(v["dir"].glob("seed_*/*")):
            local, kd = sdir / "local" / "predictions", sdir / "heterogeneous_kd" / "predictions"
            for phase, d, prefix in (("local (ensemble members)", local, "bank_"),
                                     ("after KD (committee members)", kd, "kd_bank_")):
                files = sorted(d.glob(f"{prefix}[A-Z].npz"))
                if len(files) < 2 or not (sdir / d.parent.name / "status.json").is_file() or \
                        read_json(sdir / d.parent.name / "status.json").get("status") != "completed":
                    continue
                rho, jac = _diversity([load_npz(f)["p_val"] for f in files])
                recs.append({"role": role, "seed": int(sdir.parent.name.removeprefix("seed_")),
                             "scenario": sdir.name, "phase": phase,
                             "mean_pairwise_spearman": rho, f"top{TOP_K}_jaccard": jac})
    return pd.DataFrame(recs)


def cost(roles: dict[str, dict]) -> pd.DataFrame:
    """Per protocol: client fitting time, fits, rounds, round time, communication."""
    recs = []
    for role, v in roles.items():
        cfg = v["experiment"]["identity"]["config"]
        R = int(cfg["federation"]["num_rounds"])
        K = len(cfg["clients"])
        m = v["metrics"]
        rounds = v["rounds"]
        specs = [("local/ensemble", "ensemble", "ensemble_equal", "primary"),
                 ("KD committee", "heterogeneous_kd", "kd_committee", "primary"),
                 ("FedAvg", "fedavg", "fedavg_global", "primary"),
                 ("FedAvg (20 epochs)", "fedavg", "fedavg_global", "fedavg_20_epochs"),
                 ("Fed. XGB bagging", "fedxgb_bagging", "fedxgb_global", "primary"),
                 ("KD committee, lambda=0", "heterogeneous_kd", "kd_committee", "kd_lambda_0.0")]
        for name, method, system, variant in specs:
            g = m[(m["method"] == method) & (m["system"] == system) & (m["variant"] == variant)]
            if g.empty:
                continue
            rr = rounds[(rounds["method"] == method) & (rounds["variant"] == variant)] \
                if not rounds.empty else rounds
            per_round = np.nan
            if not rr.empty and "round_seconds" in rr:
                per_round = rr[rr["round"] >= 1].drop_duplicates(
                    ["seed", "scenario", "round"])["round_seconds"].astype(float).mean()
            fit_by_family = {}
            if method == "heterogeneous_kd" and not rr.empty:
                fr = rr[(rr["round"] >= 1) & rr["system"].str.match(r"kd_bank_[A-Z]$")]
                fit_by_family = fr.groupby("model_type")["fit_seconds"].mean().to_dict()
            fits = {"ensemble": K, "heterogeneous_kd": K * (R + 1), "fedavg": K * R,
                    "fedxgb_bagging": K * R}[method]
            fed_rounds = 0 if method == "ensemble" else R
            recs.append({"role": role, "protocol": name, "n_runs": len(g),
                         "client_fit_seconds_mean": g["train_seconds"].mean(),
                         "client_fit_seconds_std": g["train_seconds"].std(ddof=1),
                         "client_fit_seconds_max": g["train_seconds"].max(),
                         "federation_wall_seconds_mean": g["federation_wall_seconds"].mean()
                         if "federation_wall_seconds" in g else np.nan,
                         "seconds_per_round_mean": per_round,
                         "fit_seconds_per_bank_round_by_family":
                             "; ".join(f"{k}={val:.1f}" for k, val in fit_by_family.items()),
                         "model_fits_per_run": fits, "communication_rounds": fed_rounds,
                         "communication_mb_mean": g["communication_bytes"].fillna(0).mean() / 1e6,
                         "test_inference_seconds_mean": g["test_inference_seconds"].mean(),
                         "peak_memory": "not measured"})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- tables
def _tex_main(summ: pd.DataFrame, ref: pd.DataFrame, scen: list[str], labels: dict,
              tdir: Path) -> None:
    order = [s.key for s in SYSTEMS]
    rows, rules, last = [], set(), None
    cols = {sc: summ[summ["scenario"] == sc].set_index("key") for sc in scen}
    present = [k for k in order if k in set(summ["key"])]
    for k in present:
        spec = next(s for s in SYSTEMS if s.key == k)
        if spec.group != last:
            rules.add(len(rows))
            last = spec.group
        rows.append([escape(spec.label)])
    for sc in scen:
        c = cols[sc]
        means = [c.loc[k, "test_average_precision_mean"] if k in c.index else np.nan for k in present]
        cells = [fmt_mean_std(c.loc[k, "test_average_precision_mean"],
                              c.loc[k, "test_average_precision_std"]) if k in c.index else "--"
                 for k in present]
        elig = [next(s for s in SYSTEMS if s.key == k).kind != "control" for k in present]
        for r, cell in zip(rows, bold_best(cells, means, eligible=elig), strict=True):
            r.append(cell)
    tex = latex_table(rows, ["System", *[escape(labels[s]) for s in scen]],
                      caption="Test AP (mean $\\pm$ std over 3 seeds; all seeds share one test "
                              "set) of the heterogeneous committee and the homogeneous baselines "
                              "on identical seed-specific partitions. Bold: highest mean per "
                              "scenario (controls excluded); differences are interpreted with "
                              "Table~\\ref{tab:cmp-paired}, not from the bold face.",
                      label="tab:cmp-main", align="l" + "r" * len(scen), group_rule_before=rules,
                      size=r"\scriptsize")
    write_table(tdir / "comparison_main", tex, summ)
    rrows = []
    for k in [s.key for s in REFERENCE] + sorted(set(ref["key"]) - {s.key for s in REFERENCE}):
        d = ref[ref["key"] == k]
        if d.empty:
            continue
        r = [escape(d["label"].iloc[0])]
        for sc in scen:
            x = d[d["scenario"] == sc]
            r.append(fmt_mean_std(x["test_average_precision_mean"].iloc[0],
                                  x["test_average_precision_std"].iloc[0]) if len(x) else "--")
        rrows.append(r)
    if rrows:
        tex = latex_table(rrows, ["Reference", *[escape(labels[s]) for s in scen]],
                          caption="Non-federated references: centralized models trained on the "
                                  "pooled client partitions (not an upper bound, outside any "
                                  "federated protocol) and local-only models (mean of the three "
                                  "banks per seed). Test AP, mean $\\pm$ std over seeds.",
                          label="tab:cmp-ref", align="l" + "r" * len(scen), size=r"\scriptsize")
        write_table(tdir / "comparison_reference", tex, ref)


def _tex_paired(pairs: pd.DataFrame, scen: list[str], labels: dict, tdir: Path, name: str,
                questions: tuple[str, ...], caption: str, tex_label: str) -> None:
    lab = {s.key: s.label for s in SYSTEMS}
    d = pairs[pairs["question"].str.startswith(questions)]
    if d.empty:
        return
    rows, rules, last = [], set(), None
    for (q, a, b), g in d.groupby(["question", "a", "b"], sort=False):
        if q != last:
            rules.add(len(rows))
            last = q
        r = [escape(f"{lab[a]} − {lab[b]}".replace(" (proposed)", ""))]
        for sc in scen:
            x = g[g["scenario"] == sc]
            if x.empty:
                r.append("--")
                continue
            x = x.iloc[0]
            r.append(f"{x['mean_diff']:+.4f} ({x['min_diff']:+.3f}, {x['max_diff']:+.3f}) "
                     f"{x['n_a_better']}/{x['n_pairs']}")
        rows.append(r)
    tex = latex_table(rows, ["Comparison (a − b)", *[escape(labels[s]) for s in scen]],
                      caption=caption, label=tex_label, align="l" + "r" * len(scen),
                      group_rule_before=rules, size=r"\scriptsize")
    write_table(tdir / name, tex, d)


def _tex_simple(df: pd.DataFrame, header: list[str], cols: list[str], fmts, caption: str,
                label: str, path: Path, group_col: str | None = None) -> None:
    if df.empty:
        return
    rows, rules, last = [], set(), None
    for _, r in df.iterrows():
        if group_col and r[group_col] != last:
            rules.add(len(rows))
            last = r[group_col]
        rows.append([f(r[c]) for c, f in zip(cols, fmts, strict=True)])
    write_table(path, latex_table(rows, header, caption=caption, label=label,
                                  align="l" * 2 + "r" * (len(header) - 2),
                                  group_rule_before=rules, size=r"\scriptsize"), df)


# --------------------------------------------------------------------------- figures
def _fig_ap(rows: pd.DataFrame, summ: pd.DataFrame, scen: list[str], labels: dict, fdir: Path,
            banner) -> list[Path]:
    plt = viz.plt
    keys = [s.key for s in SYSTEMS if s.key in set(summ["key"])]
    specs = {s.key: s for s in SYSTEMS}
    fig, axes = plt.subplots(1, len(scen), figsize=(2.35 * len(scen) + 2.6, 0.27 * len(keys) + 1.4),
                             squeeze=False, sharey=True, sharex=True)
    y = np.arange(len(keys))[::-1]
    for j, sc in enumerate(scen):
        ax = axes[0][j]
        for yi, k in zip(y, keys, strict=True):
            s = specs[k]
            pts = rows[(rows["key"] == k) & (rows["scenario"] == sc)]["test_average_precision"]
            x = summ[(summ["key"] == k) & (summ["scenario"] == sc)]
            if x.empty:
                continue
            col = FAMILY_COLORS[s.family]
            ax.scatter(pts, np.full(len(pts), yi), s=10, color=viz.TEXT_2, alpha=0.45, zorder=2,
                       linewidths=0)
            m, sd = x["test_average_precision_mean"].iloc[0], x["test_average_precision_std"].iloc[0]
            ax.errorbar(m, yi, xerr=0 if np.isnan(sd) else sd, fmt=KIND_MARKERS[s.kind],
                        color=col, markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                        elinewidth=1.2, capsize=2, zorder=3,
                        markerfacecolor=col if s.kind != "ensemble" else "white")
        ax.set_title(labels[sc])
        ax.set_xlabel("Test AP")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(y, [specs[k].label for k in keys])
    lo = np.nanmin(rows["test_average_precision"]) - 0.02
    hi = np.nanmax(rows["test_average_precision"]) + 0.01
    for ax in axes[0]:
        ax.set_xlim(lo, hi)
    fam = [plt.Line2D([], [], color=FAMILY_COLORS[f], marker="o", linestyle="", markersize=7)
           for f in FAMILY_COLORS]
    kinds = [k for k in ("kd", "ensemble", "fl", "control") if k in {specs[x].kind for x in keys}]
    kh = [plt.Line2D([], [], color=viz.TEXT_2, marker=KIND_MARKERS[k], linestyle="",
                     markersize=7, markerfacecolor="white" if k == "ensemble" else viz.TEXT_2)
          for k in kinds]
    fig.legend(fam + kh, [FAMILY_NAMES[f] for f in FAMILY_COLORS] + [KIND_NAMES[k] for k in kinds],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, -0.02))
    fig.text(0.99, 0.01, "grey dots: individual seeds; marker ± bar: mean ± std",
             ha="right", fontsize=6.5, color=viz.TEXT_2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return viz._save(fig, fdir, "fig_c1_ap_by_system", banner)


def _fig_paired(pairs: pd.DataFrame, boot: pd.DataFrame, scen: list[str], labels: dict,
                fdir: Path, banner) -> list[Path]:
    plt = viz.plt
    d = pairs[pairs["a"] == PROPOSED]
    if d.empty:
        return []
    specs = {s.key: s for s in SYSTEMS}
    bases = [s.key for s in SYSTEMS if s.key in set(d["b"])]
    fig, axes = plt.subplots(1, len(scen), figsize=(2.35 * len(scen) + 2.6,
                                                    0.3 * len(bases) + 1.3),
                             squeeze=False, sharey=True, sharex=True)
    y = np.arange(len(bases))[::-1]
    for j, sc in enumerate(scen):
        ax = axes[0][j]
        ax.axvline(0, color=viz.TEXT_2, linewidth=0.9, zorder=1)
        for yi, b in zip(y, bases, strict=True):
            bs = boot[(boot["a"] == PROPOSED) & (boot["b"] == b) & (boot["scenario"] == sc)]
            for _, r in bs.iterrows():  # per-seed paired bootstrap interval (test resampling)
                ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=viz.GRID, linewidth=3.2,
                        solid_capstyle="round", zorder=1.5)
            ax.scatter(bs["diff"], np.full(len(bs), yi), s=12, color=viz.TEXT_2, zorder=2,
                       linewidths=0)
            x = d[(d["b"] == b) & (d["scenario"] == sc)]
            if len(x):
                ax.plot(x["mean_diff"].iloc[0], yi, marker="|", markersize=13, markeredgewidth=2.2,
                        color=FAMILY_COLORS[specs[b].family], zorder=3)
        ax.set_title(labels[sc])
        ax.set_xlabel("AP(proposed) − AP(baseline)")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(y, [specs[b].label for b in bases])
    fig.text(0.99, 0.005, "bar: mean over seeds · dots: seeds · grey band: per-seed paired "
             "bootstrap 95% interval (test resampling)", ha="right", fontsize=6.5,
             color=viz.TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return viz._save(fig, fdir, "fig_c2_paired_differences", banner)


def _fig_cost(costdf: pd.DataFrame, summ: pd.DataFrame, fdir: Path, banner) -> list[Path]:
    """Mean AP over scenarios vs. client fitting time per run (log scale), one point per system."""
    plt = viz.plt
    link = {("hetero", "KD committee"): "kd_het", ("hetero", "local/ensemble"): "ens_het",
            ("hetero", "FedAvg"): "fedavg", ("mlp", "FedAvg (20 epochs)"): "fedavg_20ep",
            ("xgboost", "Fed. XGB bagging"): "fedxgb",
            **{(r, "KD committee"): f"kd_{s}" for r, s in
               (("xgboost", "xgb"), ("random_forest", "rf"), ("mlp", "mlp"))},
            **{(r, "local/ensemble"): f"ens_{s}" for r, s in
               (("xgboost", "xgb"), ("random_forest", "rf"), ("mlp", "mlp"))}}
    specs = {s.key: s for s in SYSTEMS}
    pts = []
    for _, r in costdf.iterrows():
        k = link.get((r["role"], r["protocol"]))
        if k is None or k not in set(summ["key"]):
            continue
        ap = summ[summ["key"] == k]["test_average_precision_mean"].mean()
        pts.append((k, r["client_fit_seconds_mean"], ap))
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for k, t, ap in pts:
        s = specs[k]
        col = FAMILY_COLORS[s.family]
        ax.plot(t, ap, marker=KIND_MARKERS[s.kind], color=col, markersize=8, linestyle="",
                markerfacecolor=col if s.kind != "ensemble" else "white", markeredgewidth=1.3)
        ax.annotate(s.label.replace(" (proposed)", "*"), (t, ap), xytext=(5, 3),
                    textcoords="offset points", fontsize=6.5, color=viz.TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Client fitting time per run (s, summed over banks; log scale)")
    ax.set_ylabel("Test AP (mean over scenarios)")
    ax.set_title("Detection performance vs. training cost")
    fig.tight_layout()
    return viz._save(fig, fdir, "fig_c3_cost_vs_ap", banner)


# --------------------------------------------------------------------------- main
def compare(dirs: dict[str, Path], out: Path) -> Path:
    unknown = set(dirs) - set(ROLES)
    if unknown:
        raise ComparisonError(f"Unknown roles {sorted(unknown)}; valid: {ROLES}")
    if "hetero" not in dirs:
        raise ComparisonError("The heterogeneous results directory (--hetero) is required")
    roles = load_roles(dirs)
    checks = verify(roles)
    out = Path(out)
    tdir, fdir, rdir = out / "tables", out / "figures", out / "reports"
    for d in (tdir, fdir, rdir):
        d.mkdir(parents=True, exist_ok=True)
    het_cfg = roles["hetero"]["experiment"]["identity"]["config"]
    rows = system_rows(roles, SYSTEMS)
    scen = order_scenarios(het_cfg, rows["scenario"].unique())
    labels = scenario_labels(het_cfg, scen)
    metrics = ["test_average_precision", *[c for c, _ in SECONDARY], *CONFUSION,
               "train_seconds", "communication_bytes"]
    metrics = [c for c in metrics if c in rows]
    summ = seed_summary(rows, metrics)
    write_csv(rdir / "per_seed_metrics.csv",
              rows[["key", "label", "group", "role", "seed", "scenario", "partition_sha256",
                    "system", "variant", *metrics, "run_dir"]])

    # references: centralized + local-only family means (mean of the 3 banks per seed)
    ref_rows = system_rows(roles, REFERENCE)
    loc = []
    for role in ("xgboost", "random_forest", "mlp"):
        if role in roles:
            m = roles[role]["metrics"]
            g = m[(m["variant"] == "primary") & m["system"].str.match(r"bank_[A-Z]$")]
            g = g.groupby(["seed", "scenario"], as_index=False)["test_average_precision"].mean()
            loc.append(g.assign(key=f"local_{role}",
                                label=f"Local {FAMILY_NAMES[role]} (mean of 3 banks)",
                                group="Reference"))
    ref_rows = pd.concat([ref_rows, *loc], ignore_index=True)
    ref = seed_summary(ref_rows, ["test_average_precision"])

    bcfg = (het_cfg.get("evaluation") or {}).get("bootstrap") or {}
    pairs, boot = paired(rows, {"n_resamples": 1000, "confidence": 0.95, "seed": 0, **bcfg})
    write_csv(rdir / "paired_differences.csv", pairs)
    write_csv(rdir / "paired_bootstrap_per_seed.csv", boot)
    pb, div, costdf, prov = per_bank(roles), diversity(roles), cost(roles), provenance(roles)
    write_json(rdir / "checks.json", checks)

    _tex_main(summ, ref, scen, labels, tdir)
    note = ("Cells: mean seed-paired difference in test AP (min, max over seeds) and the "
            "number of seeds in which a is higher. Seeds share one test set; per-seed paired "
            "bootstrap intervals are in reports/paired\\_bootstrap\\_per\\_seed.csv. "
            "Descriptive only: no significance tests.")
    _tex_paired(pairs, scen, labels, tdir, "comparison_paired",
                ("Q1", "Q2"), "Proposed heterogeneous KD committee minus each homogeneous "
                "baseline. " + note, "tab:cmp-paired")
    _tex_paired(pairs, scen, labels, tdir, "comparison_mechanism", ("Q3", "Q4"),
                "Mechanism comparisons: architectural heterogeneity without KD (Q3) and KD "
                "versus the matched committee without KD (Q4). " + note, "tab:cmp-mechanism")
    sec = summ[["key", "label", "scenario", *[f"{c}_mean" for c, _ in SECONDARY]]].copy()
    sec["scenario_label"] = sec["scenario"].map(labels)
    sec = sec.sort_values("scenario", key=lambda s: s.map({x: i for i, x in enumerate(scen)}),
                          kind="stable")
    _tex_simple(sec, ["Scenario", "System", *[h for _, h in SECONDARY]],
                ["scenario_label", "label", *[f"{c}_mean" for c, _ in SECONDARY]],
                [escape, escape] + [lambda v: f"{v:.3f}"] * 4 + [lambda v: f"{v:.1f}"] * 2,
                "Secondary test metrics (mean over seeds). Precision, recall, F1, FP and FN "
                "use each system's validation-selected (max-F1) threshold; 98 test frauds.",
                "tab:cmp-secondary", tdir / "comparison_secondary", group_col="scenario_label")
    if not pb.empty:
        pbt = pb.copy()
        pbt["committee"] = pbt["role"].map(ROLE_LABELS)
        pbt["scenario_label"] = pbt["scenario"].map(labels)
        pbt["bank_lbl"] = pbt["bank"] + " (" + pbt["model_type"] + ")"
        _tex_simple(pbt, ["Committee", "Scenario", "Bank", "Before", "After", "$\\Delta$ (min, max)",
                          "Improved"],
                    ["committee", "scenario_label", "bank_lbl", "before_mean", "after_mean",
                     "delta_mean", "n_improved"],
                    [escape, escape, escape, lambda v: f"{v:.3f}", lambda v: f"{v:.3f}",
                     lambda v: f"{v:+.4f}", str],
                    "Per-bank test AP before (local-only) and after KD (mean over seeds; "
                    "$\\Delta$ seed-paired).", "tab:cmp-perbank", tdir / "comparison_per_bank",
                    group_col="committee")
    if not div.empty:
        dv = div.groupby(["role", "scenario", "phase"], as_index=False)[
            ["mean_pairwise_spearman", f"top{TOP_K}_jaccard"]].mean()
        dv["committee"] = dv["role"].map(ROLE_LABELS)
        dv["scenario_label"] = dv["scenario"].map(labels)
        write_csv(rdir / "diversity_per_seed.csv", div)
        _tex_simple(dv, ["Committee", "Scenario", "Members", "Rank corr.", f"Top-{TOP_K} overlap"],
                    ["committee", "scenario_label", "phase", "mean_pairwise_spearman",
                     f"top{TOP_K}_jaccard"],
                    [escape, escape, escape, lambda v: f"{v:.3f}", lambda v: f"{v:.3f}"],
                    "Member agreement on the validation split (mean over seeds): mean pairwise "
                    "Spearman correlation of the members' fraud scores and mean pairwise Jaccard "
                    f"overlap of their {TOP_K} highest-scored transactions. Lower = more diverse "
                    "committee. Diagnostic only; not used for any selection.",
                    "tab:cmp-diversity", tdir / "comparison_diversity", group_col="committee")
    if not costdf.empty:
        cd = costdf.copy()
        cd["committee"] = cd["role"].map(ROLE_LABELS)
        _tex_simple(cd, ["Profile", "Protocol", "Fit (s)", "s/round", "Fits/run", "Rounds",
                         "Comm. (MB)"],
                    ["committee", "protocol", "client_fit_seconds_mean", "seconds_per_round_mean",
                     "model_fits_per_run", "communication_rounds", "communication_mb_mean"],
                    [escape, escape, lambda v: f"{v:.1f}",
                     lambda v: "--" if pd.isna(v) else f"{v:.1f}", str, str,
                     lambda v: f"{v:.2f}"],
                    "Cost per run (mean over seeds and scenarios). Fit = summed client fitting "
                    "time (banks run in parallel in the simulation); s/round = server-observed "
                    "round time. Communication counts serialised payloads in both directions "
                    "(federated XGBoost: serialised tree size, estimated). The no-KD ensembles "
                    "exchange nothing during training but need every bank's model (or a scoring "
                    "service) at inference. Peak memory was not measured.",
                    "tab:cmp-cost", tdir / "comparison_cost", group_col="committee")

    cfg_pub = all(prov["all_from_clean_commit"])
    banner = None if cfg_pub else "DEVELOPMENT RESULTS - source not a clean commit"
    figs = {"fig_c1": _fig_ap(rows, summ, scen, labels, fdir, banner),
            "fig_c2": _fig_paired(pairs, boot, scen, labels, fdir, banner),
            "fig_c3": _fig_cost(costdf, summ, fdir, banner)}
    _summary(out, rdir, checks, prov, summ, ref, pairs, pb, div, costdf, scen, labels, figs)
    return rdir / "summary.md"


def _md(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def _summary(out, rdir, checks, prov, summ, ref, pairs, pb, div, costdf, scen, labels, figs):
    L = ["# Heterogeneous vs homogeneous committees", ""]
    if not prov["all_from_clean_commit"].all():
        L += ["> **Development results.** Not every run was produced from a clean git commit "
              "(see Provenance); the exact source of each run is in its "
              "`invocations/<id>/source.patch`. Re-run from a commit with "
              "`--require-clean-source` before publication.", ""]
    L += ["## Comparability checks (all passed, otherwise no report is written)", "",
          f"- Dataset SHA-256 `{checks['dataset_sha256']}`, split bundle `{checks['split_tag']}`",
          f"- Client model families: " + "; ".join(f"{r}: {'/'.join(v)}"
                                                   for r, v in checks["clients"].items()),
          "- Semantic configurations differ only in `clients`, `profile`, `fedxgb`",
          f"- Identical partitions and test labels in all {checks['n_seed_scenarios']} "
          "seed × scenario cells", "- Library versions: " + ", ".join(
              f"{k} {v}" for k, v in checks["packages"].items() if v), "",
          "## Provenance", "", _md(prov, 2), ""]
    cols = [s for s in scen if s in set(summ["scenario"])]
    piv = summ.pivot_table(index="label", columns="scenario", values="test_average_precision_mean",
                           sort=False, dropna=False).reindex(columns=cols)
    sd = summ.pivot_table(index="label", columns="scenario", values="test_average_precision_std",
                          sort=False, dropna=False).reindex(index=piv.index, columns=cols)
    cell = piv.map(lambda v: f"{v:.4f}") + " ± " + sd.map(lambda v: f"{v:.4f}")
    L += ["## Test AP (mean ± std over seeds)", "", _md(cell.reset_index().rename(columns=labels)),
          "", "Seeds share one test set: the spread is training variability (partition + "
          "initialisation), not test-sample uncertainty.", ""]
    rp = ref.pivot_table(index="label", columns="scenario", values="test_average_precision_mean",
                         sort=False)
    L += ["### Non-federated references", "",
          _md(rp[[s for s in scen if s in rp.columns]].reset_index().rename(columns=labels)), ""]
    if not pairs.empty:
        lab = {s.key: s.label for s in SYSTEMS}
        p = pairs.assign(a=pairs["a"].map(lab), b=pairs["b"].map(lab),
                         scenario=pairs["scenario"].map(labels))
        L += ["## Seed-paired AP differences (a − b), descriptive", "",
              _md(p[["question", "a", "b", "scenario", "n_pairs", "mean_diff", "std_diff",
                     "min_diff", "max_diff", "relative_diff_pct", "n_a_better",
                     "n_seeds_ci_above_0", "n_seeds_ci_below_0"]]), "",
              "`relative_diff_pct` = mean difference / mean AP of b × 100 (relative, not "
              "absolute). `n_seeds_ci_*` count seeds whose paired-bootstrap 95% interval "
              "(test resampling, one trained pair per seed) lies entirely above/below zero; "
              "intervals of different seeds are not independent replications and no "
              "multiple-comparison correction is applied. Nothing here is a significance test.",
              ""]
    if not pb.empty:
        L += ["## Per-bank AP before (local-only) and after KD", "",
              _md(pb[["role", "scenario", "bank", "model_type", "before_mean", "after_mean",
                      "delta_mean", "delta_min", "delta_max", "n_improved", "n_seeds"]]), ""]
    if not div.empty:
        dv = div.groupby(["role", "scenario", "phase"], as_index=False)[
            ["mean_pairwise_spearman", f"top{TOP_K}_jaccard"]].mean()
        L += ["## Member diversity (validation split; lower = more diverse)", "", _md(dv, 3), ""]
    if not costdf.empty:
        L += ["## Cost per run", "", _md(costdf, 2), ""]
    L += ["## Figures", ""] + [f"- {k}: " + (", ".join(Path(x).name for x in v) if v else "none")
                               for k, v in figs.items()]
    L += ["", "## Tables", ""] + [f"- `{x.name}`" for x in sorted((out / "tables").glob("*.tex"))]
    (rdir / "summary.md").write_text("\n".join(L) + "\n")
