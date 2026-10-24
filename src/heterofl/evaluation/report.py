"""Build tables, figures and statistical summaries from a results directory.

    python scripts/generate_report.py --results results/full

Only runs whose ``status.json`` says ``completed`` contribute numbers. Failed
or missing runs are listed in ``reports/run_status.csv`` and in the summary;
they are never imputed. Outputs go to ``<results>/tables``, ``<results>/figures``
and ``<results>/reports``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from heterofl.evaluation import visualization as viz
from heterofl.evaluation.latex_tables import (
    bold_best,
    escape,
    fmt_mean_std,
    fmt_num,
    latex_table,
    write_table,
)
from heterofl.evaluation.statistics import (
    bootstrap_ap_ci,
    paired_bootstrap_ap_diff,
    paired_seed_differences,
    summarize_across_seeds,
)
from heterofl.utils.io import load_npz, read_json, write_csv
from heterofl.utils.logging import get_logger

log = get_logger("report")

MODEL_SHORT = {"xgboost": "XGB", "random_forest": "RF", "mlp": "MLP", "committee": "committee"}
SYSTEM_ORDER = ["bank_A", "bank_B", "bank_C", "centralized_xgboost", "centralized_random_forest",
                "centralized_mlp", "ensemble_equal", "ensemble_validation_ap", "fedavg_global",
                "kd_bank_A", "kd_bank_B", "kd_bank_C", "kd_committee"]
METRICS = ["test_average_precision", "test_pr_auc_trapezoid", "test_roc_auc",
           "test_at_val_threshold_precision", "test_at_val_threshold_recall",
           "test_at_val_threshold_f1", "test_at_val_threshold_balanced_accuracy",
           "test_at_val_threshold_fpr", "test_at_val_threshold_fp", "test_at_val_threshold_fn",
           "test_at_0.5_precision", "test_at_0.5_recall", "test_at_0.5_f1",
           "test_recall_at_fpr_0.001", "test_recall_at_fpr_0.01",
           "validation_average_precision", "validation_threshold",
           "train_seconds", "test_inference_seconds", "communication_bytes"]
COMPARISONS = [("kd_committee", "ensemble_equal"), ("kd_committee", "fedavg_global"),
               ("ensemble_equal", "fedavg_global"), ("kd_bank_A", "bank_A"),
               ("kd_bank_B", "bank_B"), ("kd_bank_C", "bank_C")]


# --------------------------------------------------------------------------- collection
def _family(row) -> str:
    s = row["system"]
    if row["method"] == "local":
        return "Local"
    if s.startswith("centralized"):
        return "Centralized (ref.)"
    if s.startswith("ensemble"):
        return "Ensemble (no KD)"
    if s == "fedavg_global":
        return "FedAvg"
    if s == "kd_committee":
        return "KD committee"
    return "KD (per bank)"


def _short(row) -> str:
    s, mt = row["system"], MODEL_SHORT.get(row["model_type"], row["model_type"])
    if s.startswith("bank_"):
        return f"Local {s[-1]} ({mt})"
    if s.startswith("centralized_"):
        return f"Central. {mt} (ref.)"
    if s == "ensemble_equal":
        return "Ensemble (equal)"
    if s == "ensemble_validation_ap":
        return "Ensemble (val-AP wtd)"
    if s == "fedavg_global":
        return "FedAvg MLP"
    if s == "kd_committee":
        return "KD committee"
    if s.startswith("kd_bank_"):
        return f"KD {s.split('_')[2]} ({mt})"
    return s


def scenario_labels(cfg: dict | None, scenarios: list[str]) -> dict[str, str]:
    out = {}
    for s in scenarios:
        sc = (cfg or {}).get("scenarios", {}).get(s) or {}
        if sc.get("type") == "iid" or s == "iid":
            out[s] = "IID"
        elif "alpha" in sc:
            out[s] = f"{s.replace('_non_iid', '').capitalize()} (α={sc['alpha']})"
        else:
            out[s] = s
    return out


def order_scenarios(cfg: dict | None, scenarios) -> list[str]:
    def key(s):
        sc = (cfg or {}).get("scenarios", {}).get(s) or {}
        return (0, 0) if sc.get("type") == "iid" or s == "iid" else (1, -float(sc.get("alpha", 0)))
    return sorted(set(scenarios), key=key)


def collect(root: Path) -> dict:
    status_rows, metrics, rounds = [], [], []
    patterns = ["seed_*/*/*/status.json", "seed_*/*/ablations/*/status.json"]
    for pat in patterns:
        for st in sorted(root.glob(pat)):
            if "_archive" in st.parts or "debug" in st.parts:
                continue
            d = st.parent
            s = read_json(st)
            is_ab = d.parent.name == "ablations"
            scen_dir = d.parent.parent if is_ab else d.parent
            ident = read_json(d / "run_identity.json") if (d / "run_identity.json").is_file() else {}
            src = ident.get("source") or {}
            status_rows.append({"path": str(d.relative_to(root)), "seed": scen_dir.parent.name,
                                "scenario": scen_dir.name, "run": d.name,
                                "ablation": is_ab, "status": s.get("status"),
                                "error": s.get("error"), "wall_seconds": s.get("wall_seconds"),
                                "legacy_no_identity": not ident,
                                "run_hash": ident.get("run_hash"),
                                "config_hash": ident.get("config_hash"),
                                "source_commit": src.get("commit"),
                                "source_dirty": src.get("dirty"),
                                "source_diff_sha256": src.get("diff_sha256"),
                                "source_verified": bool(src.get("source_verified"))})
            if s.get("status") != "completed":
                continue
            if (d / "metrics.csv").is_file():
                m = pd.read_csv(d / "metrics.csv")
                m["run_dir"] = str(d)
                metrics.append(m)
            if (d / "rounds.csv").is_file():
                r = pd.read_csv(d / "rounds.csv")
                r["run_dir"] = str(d)
                rounds.append(r)
    parts = [pd.read_csv(p) for p in sorted(root.glob("seed_*/*/partition_report.csv"))]
    mdf = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    if not mdf.empty:
        if "kd_phase" not in mdf:
            mdf["kd_phase"] = np.nan
        mdf["family"] = mdf.apply(_family, axis=1)
        mdf["short_label"] = mdf.apply(_short, axis=1)
    return {"status": pd.DataFrame(status_rows), "metrics": mdf,
            "rounds": pd.concat(rounds, ignore_index=True) if rounds else pd.DataFrame(),
            "partitions": pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()}


def provenance_summary(root: Path, status: pd.DataFrame) -> dict:
    """Is every completed run traceable to one configuration and a clean source commit?"""
    exp = read_json(root / "experiment.json") if (root / "experiment.json").is_file() else {}
    done = status[status["status"] == "completed"] if not status.empty else status
    problems = []
    if not exp:
        problems.append("legacy results: no experiment identity recorded (configuration "
                        "hash and source provenance cannot be verified)")
    if not done.empty:
        if done["legacy_no_identity"].any():
            problems.append(f"{int(done['legacy_no_identity'].sum())} run(s) without a recorded "
                            "run identity")
        if not done["source_verified"].all():
            problems.append(f"{int((~done['source_verified']).sum())} run(s) not produced from "
                            "a clean git commit")
        srcs = done[["source_commit", "source_dirty", "source_diff_sha256"]].drop_duplicates()
        if len(srcs) > 1:
            problems.append(f"runs come from {len(srcs)} different source revisions")
        sources_md = "; ".join(
            f"`{(r.source_commit or 'unknown')[:12]}`"
            f"{' (dirty, diff ' + str(r.source_diff_sha256)[:12] + ')' if r.source_dirty else ''}"
            for r in srcs.itertuples())
    else:
        sources_md = "none"
    return {"experiment_hash": exp.get("experiment_hash", "none (legacy)"),
            "config_hash": exp.get("config_hash", "none (legacy)"),
            "sources_md": sources_md, "problems": problems, "verified": not problems}


def _primary(mdf: pd.DataFrame) -> pd.DataFrame:
    if mdf.empty:
        return mdf
    d = mdf[(mdf["variant"] == "primary") & (mdf["kd_phase"].fillna("after") != "before")]
    return d[d["system"].isin(SYSTEM_ORDER)]


def _sort_systems(df: pd.DataFrame) -> pd.DataFrame:
    order = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    return df.assign(_o=df["system"].map(order)).sort_values(["_o"]).drop(columns="_o")


# --------------------------------------------------------------------------- tables
def table1(root: Path, parts: pd.DataFrame, scen: list[str], labels: dict, tdir: Path) -> pd.DataFrame:
    split = read_json(root / "split_manifest.json") if (root / "split_manifest.json").is_file() else None
    rows = []
    if split:
        for name, s in split["splits"].items():
            rows.append({"section": "split", "name": name, "n": s["n"], "fraud": s["fraud"],
                         "fraud_pct": 100 * s["fraud_rate"]})
    het = []
    for (sc, seed), g in parts.groupby(["scenario", "partition_seed"]):
        fr = g["fraud_pct"].to_numpy()
        het.append({"scenario": sc, "partition_seed": seed,
                    "fraud_rate_cv": float(fr.std(ddof=0) / fr.mean()) if fr.mean() else float("nan"),
                    "fraud_rate_max_over_min": float(fr.max() / fr.min()) if fr.min() > 0 else float("inf"),
                    "largest_client_share": float(g["relative_size"].max())})
    hdf = pd.DataFrame(het)
    full = parts.merge(hdf, on=["scenario", "partition_seed"], how="left") if not parts.empty else parts
    write_csv(tdir / "table1_partitions.csv", full)
    if rows:
        write_csv(tdir / "table1_splits.csv", rows)

    if rows:
        names = {"test": "Test", "validation": "Validation",
                 "proxy": "Proxy / public reference (labels withheld)",
                 "client_pool": "Client pool (partitioned among banks)"}
        split_rows = [[escape(names.get(r["name"], r["name"])), fmt_num(r["n"]), fmt_num(r["fraud"]),
                       f"{r['fraud_pct']:.3f}"] for r in rows]
        tex = latex_table(split_rows, ["Split", "Transactions", "Frauds", "Fraud \\%"],
                          caption="Fixed, mutually disjoint splits (identical for every seed, "
                                  "scenario and condition).", label="tab:splits", align="lrrr")
        write_table(tdir / "table1a_splits", tex)

    tex_rows, rules = [], set()
    for sc in scen:
        g0 = full[full["scenario"] == sc]
        rules.add(len(tex_rows))
        for i, (seed, g) in enumerate(g0.groupby("partition_seed")):
            cells = [f"{int(b['n_transactions']):,} / {int(b['n_fraud'])} ({b['fraud_pct']:.2f}\\%)"
                     .replace(",", "{,}") for _, b in g.sort_values("bank").iterrows()]
            h = hdf[(hdf["scenario"] == sc) & (hdf["partition_seed"] == seed)].iloc[0]
            tex_rows.append([escape(labels.get(sc, sc)) if i == 0 else "", str(seed), *cells,
                             f"{h['fraud_rate_cv']:.2f}", f"{h['largest_client_share']:.2f}"])
    if tex_rows:
        tex = latex_table(tex_rows, ["Scenario", "Seed", "Bank A (XGB)", "Bank B (RF)", "Bank C (MLP)",
                                     "CV(fraud rate)", "Max share"],
                          caption=("Client partitions as realised, shown as ``transactions / frauds "
                                   "(fraud rate)''. CV is the coefficient of variation of the three "
                                   "clients' fraud rates; max share is the largest client's fraction "
                                   "of the client pool. Banks are simulated partitions of the ULB "
                                   "data, not real institutions."),
                          label="tab:data", align="llrrrrr", group_rule_before=rules,
                          size=r"\scriptsize")
        write_table(tdir / "table1b_partitions", tex)
    return hdf


def _group_table(summ: pd.DataFrame, scen: list[str], labels: dict, cols: list[tuple[str, str]],
                 systems_filter, caption: str, label: str, tdir: Path, name: str,
                 exclude_from_bold=lambda s: s.startswith("centralized")) -> None:
    rows, rules = [], set()
    for sc in scen:
        d = _sort_systems(summ[(summ["scenario"] == sc) & summ["system"].apply(systems_filter)])
        if d.empty:
            continue
        rules.add(len(rows))
        block = [[escape(labels.get(sc, sc)) if i == 0 else "", escape(r["short_label"])]
                 for i, (_, r) in enumerate(d.iterrows())]
        eligible = [not exclude_from_bold(s) for s in d["system"]]
        for key, _ in cols:
            cells = [fmt_mean_std(r[f"{key}_mean"], r[f"{key}_std"]) for _, r in d.iterrows()]
            cells = bold_best(cells, d[f"{key}_mean"].tolist(), eligible=eligible)
            for b, c in zip(block, cells, strict=True):
                b.append(c)
        rows.extend(block)
    if not rows:
        return
    tex = latex_table(rows, ["Scenario", "System", *[h for _, h in cols]], caption=caption,
                      label=label, align="ll" + "r" * len(cols), group_rule_before=rules,
                      size=r"\scriptsize")
    write_table(tdir / name, tex, summ[summ["system"].apply(systems_filter)])


def tables_main(summ: pd.DataFrame, scen_primary: list[str], scen_all: list[str], labels: dict,
                tdir: Path) -> None:
    n_lo = int(summ["test_average_precision_n"].min()) if not summ.empty else 0
    n_hi = int(summ["test_average_precision_n"].max()) if not summ.empty else 0
    n_txt = f"{n_hi}" if n_lo == n_hi else f"{n_lo}--{n_hi} (see the CSV for per-cell n)"
    ms = f"mean $\\pm$ std over {n_txt} seed(s); all seeds share one test set"
    _group_table(summ, scen_all, labels,
                 [("test_average_precision", "AP"), ("test_roc_auc", "ROC-AUC"),
                  ("test_at_val_threshold_f1", "F1"), ("test_recall_at_fpr_0.001", "Rec@FPR=0.1\\%")],
                 lambda s: s.startswith("bank_"),
                 f"Local-only models (Experiment 1), test split, {ms}. F1 uses the "
                 "validation-selected threshold.", "tab:local", tdir, "table2_local")
    _group_table(summ, scen_primary, labels,
                 [("test_average_precision", "AP"), ("test_pr_auc_trapezoid", "PR-AUC (trap.)"),
                  ("test_roc_auc", "ROC-AUC"), ("test_at_val_threshold_precision", "Prec."),
                  ("test_at_val_threshold_recall", "Rec."), ("test_at_val_threshold_f1", "F1"),
                  ("test_recall_at_fpr_0.001", "Rec@FPR=0.1\\%")],
                 lambda s: s.startswith(("centralized", "fedavg", "ensemble_equal", "kd_")),
                 f"Centralized reference, FedAvg, heterogeneous ensemble (no KD) and heterogeneous "
                 f"KD, test split, {ms}. AP = average precision (step-wise), PR-AUC (trap.) = "
                 "trapezoidal area. Threshold-dependent metrics use the validation-selected "
                 "threshold (max F1). Bold: best per scenario among federated/local systems; the "
                 "centralized rows are a non-federated reference and are not bolded.",
                 "tab:main", tdir, "table3_main")

    rows = []
    systems = ["bank_A", "bank_B", "bank_C", "ensemble_equal", "fedavg_global", "kd_bank_A",
               "kd_bank_B", "kd_bank_C", "kd_committee", "centralized_xgboost"]
    for s in systems:
        d = summ[summ["system"] == s]
        if d.empty:
            continue
        r = [escape(d["short_label"].iloc[0])]
        for sc in scen_all:
            x = d[d["scenario"] == sc]
            r.append(fmt_mean_std(x["test_average_precision_mean"].iloc[0],
                                  x["test_average_precision_std"].iloc[0]) if len(x) else "--")
        rows.append(r)
    if rows:
        tex = latex_table(rows, ["System", *[escape(labels.get(s, s)) for s in scen_all]],
                          caption=f"Test AP under increasing client heterogeneity ({ms}). "
                                  "Dirichlet $\\alpha$ does not fix the realised heterogeneity; "
                                  "see Table~\\ref{tab:data} for the achieved distributions.",
                          label="tab:noniid", align="l" + "r" * len(scen_all), size=r"\scriptsize")
        write_table(tdir / "table4_non_iid", tex,
                    summ[summ["system"].isin(systems)][["scenario", "system", "short_label",
                                                        "test_average_precision_mean",
                                                        "test_average_precision_std",
                                                        "test_average_precision_n"]])

    _group_table(summ, scen_all, labels,
                 [("test_average_precision", "AP"), ("test_roc_auc", "ROC-AUC"),
                  ("test_at_val_threshold_f1", "F1"), ("validation_average_precision", "Val. AP")],
                 lambda s: True, f"All systems, {ms}. Validation AP is shown for transparency; it "
                 "was used for threshold selection only.", "tab:seeds", tdir, "table7_seed_summary",
                 exclude_from_bold=lambda s: True)


def table5(prim: pd.DataFrame, scen: list[str], labels: dict, tdir: Path) -> pd.DataFrame:
    rows, out = [], []
    for sc in scen:
        for bank in ("A", "B", "C"):
            b = prim[(prim["scenario"] == sc) & (prim["system"] == f"bank_{bank}")]
            a = prim[(prim["scenario"] == sc) & (prim["system"] == f"kd_bank_{bank}")]
            j = b.set_index("seed")["test_average_precision"].to_frame("before").join(
                a.set_index("seed")["test_average_precision"].to_frame("after"), how="inner")
            if j.empty:
                continue
            delta = j["after"] - j["before"]
            mt = b["model_type"].iloc[0]
            rec = {"scenario": sc, "bank": bank, "model_type": mt, "n_seeds": len(j),
                   "before_mean": j["before"].mean(), "before_std": j["before"].std(ddof=1),
                   "after_mean": j["after"].mean(), "after_std": j["after"].std(ddof=1),
                   "delta_mean": delta.mean(), "delta_std": delta.std(ddof=1),
                   "delta_min": delta.min(), "delta_max": delta.max(),
                   "n_improved": int((delta > 0).sum())}
            out.append(rec)
            rows.append([escape(labels.get(sc, sc)) if bank == "A" else "", f"{bank} ({MODEL_SHORT[mt]})",
                         fmt_mean_std(rec["before_mean"], rec["before_std"]),
                         fmt_mean_std(rec["after_mean"], rec["after_std"]),
                         ("+" if rec["delta_mean"] >= 0 else "") +
                         fmt_mean_std(rec["delta_mean"], rec["delta_std"]),
                         f"{rec['n_improved']}/{rec['n_seeds']}"])
    df = pd.DataFrame(out)
    if rows:
        tex = latex_table(rows, ["Scenario", "Bank", "Before (local)", "After (KD)", "$\\Delta$ AP",
                                 "Seeds improved"],
                          caption="Per-bank test AP before (local-only) and after heterogeneous "
                                  "KD; $\\Delta$ is the seed-paired difference (mean $\\pm$ std). "
                                  "Degradations are reported as observed.",
                          label="tab:perbank", align="llrrrr",
                          group_rule_before={i for i in range(0, len(rows), 3)}, size=r"\scriptsize")
        write_table(tdir / "table5_per_bank", tex, df)
    return df


def table6(mdf: pd.DataFrame, parts: pd.DataFrame, tdir: Path) -> pd.DataFrame:
    prim = mdf[(mdf["variant"] == "primary") & (mdf["kd_phase"].fillna("after") != "before")]
    rows = []
    for (seed, sc, method), g in prim.groupby(["seed", "scenario", "method"]):
        if method == "heterogeneous_kd":
            g = g[g["system"] == "kd_committee"]
            train, infer = g["train_seconds"].sum(), g["test_inference_seconds"].sum()
            comm = g["communication_bytes"].sum()
            wall = g.get("federation_wall_seconds", pd.Series([np.nan])).iloc[0]
        else:
            train, infer = g["train_seconds"].sum(), g["test_inference_seconds"].sum()
            comm = g["communication_bytes"].fillna(0).sum() if method != "centralized" else np.nan
            wall = (g["federation_wall_seconds"].iloc[0]
                    if "federation_wall_seconds" in g and method == "fedavg" else np.nan)
        rows.append({"seed": seed, "scenario": sc, "method": method, "train_seconds": train,
                     "test_inference_seconds": infer, "communication_mb": comm / 1e6,
                     "federation_wall_seconds": wall})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    labels = {"local": ("Local (3 banks)", "Local"),
              "centralized": ("Centralized (3 models, ref.)", "Centralized (ref.)"),
              "ensemble": ("Ensemble (no KD)", "Ensemble (no KD)"),
              "fedavg": ("FedAvg", "FedAvg"), "heterogeneous_kd": ("Heterogeneous KD", "KD committee")}
    num = ["train_seconds", "test_inference_seconds", "communication_mb", "federation_wall_seconds"]
    agg = df.groupby("method")[num].agg(["mean", "std"])
    out = []
    for m in ["local", "centralized", "ensemble", "fedavg", "heterogeneous_kd"]:
        if m not in agg.index:
            continue
        a = agg.loc[m]
        out.append({"method": m, "label": labels[m][0], "family": labels[m][1],
                    **{f"{k}_{s}": a[(k, s)] for k in ["train_seconds", "test_inference_seconds",
                                                      "communication_mb", "federation_wall_seconds"]
                       for s in ("mean", "std")}})
    odf = pd.DataFrame(out)
    raw_mb = None
    if not parts.empty:
        pooled = parts.groupby(["scenario", "partition_seed"])["n_transactions"].sum().mean()
        raw_mb = pooled * 31 * 4 / 1e6
    rows_tex = []
    for _, r in odf.iterrows():
        comm = ("raw data" if r["method"] == "centralized"
                else fmt_mean_std(r["communication_mb_mean"], r["communication_mb_std"], 2))
        rows_tex.append([escape(r["label"]), fmt_mean_std(r["train_seconds_mean"], r["train_seconds_std"], 1),
                         fmt_mean_std(r["test_inference_seconds_mean"], r["test_inference_seconds_std"], 2),
                         comm, fmt_mean_std(r["federation_wall_seconds_mean"],
                                            r["federation_wall_seconds_std"], 1)])
    note = (f"Centralized training would require pooling the raw client rows "
            f"($\\approx${raw_mb:.1f} MB as float32), which is outside any federated protocol. "
            if raw_mb else "")
    tex = latex_table(rows_tex, ["Method", "Train (s)", "Test inference (s)", "Comm. (MB)",
                                 "Federation wall (s)"],
                      caption="Computational and communication overhead per run (mean $\\pm$ std "
                              "over seeds and scenarios). Train = summed client fitting time "
                              "(clients run in parallel in the simulation). Communication counts "
                              "serialised array payloads in both directions. " + note +
                              "Random Forest is fully retrained every KD round.",
                      label="tab:cost", align="lrrrr", size=r"\scriptsize")
    write_table(tdir / "table6_cost", tex, odf)
    return odf


def table_ablations(mdf: pd.DataFrame, scen: list[str], labels: dict, tdir: Path) -> pd.DataFrame:
    kd = mdf[(mdf["method"] == "heterogeneous_kd") & (mdf["kd_phase"].fillna("after") == "after")]
    ens = mdf[mdf["method"] == "ensemble"]
    if kd.empty:
        return pd.DataFrame()
    recs = []
    for (variant, sc, seed), g in kd.groupby(["variant", "scenario", "seed"]):
        com = g[g["system"] == "kd_committee"]["test_average_precision"]
        banks = g[g["system"].str.match(r"kd_bank_[A-Z]$")]["test_average_precision"]
        recs.append({"variant": variant, "scenario": sc, "seed": seed,
                     "kd_committee_ap": com.iloc[0] if len(com) else np.nan,
                     "kd_bank_mean_ap": banks.mean() if len(banks) else np.nan})
    for (variant, sc, seed), g in ens.groupby(["variant", "scenario", "seed"]):
        recs.append({"variant": f"ensemble:{variant}", "scenario": sc, "seed": seed,
                     "kd_committee_ap": g["test_average_precision"].iloc[0], "kd_bank_mean_ap": np.nan})
    df = pd.DataFrame(recs)
    summ = df.groupby(["variant", "scenario"]).agg(
        committee_mean=("kd_committee_ap", "mean"), committee_std=("kd_committee_ap", "std"),
        bank_mean=("kd_bank_mean_ap", "mean"), bank_std=("kd_bank_mean_ap", "std"),
        n=("seed", "nunique")).reset_index()
    write_csv(tdir / "table8_ablations.csv", summ)
    names = {"primary": "KD primary ($\\lambda$=0.5, LOO, equal)",
             "ensemble:primary": "Ensemble, equal weights (no KD)",
             "ensemble:ensemble_val_weighted": "Ensemble, val-AP weights (no KD)"}
    order = (["primary"] + sorted(v for v in summ["variant"].unique()
                                  if v != "primary" and not v.startswith("ensemble"))
             + ["ensemble:primary", "ensemble:ensemble_val_weighted"])
    rows = []
    for v in order:
        d = summ[summ["variant"] == v]
        if d.empty:
            continue
        r = [names.get(v, escape(v.replace("kd_", "KD ")))]
        for sc in scen:
            x = d[d["scenario"] == sc]
            if x.empty:
                r += ["--", "--"]
                continue
            x = x.iloc[0]
            r += [fmt_mean_std(x["committee_mean"], x["committee_std"]),
                  fmt_mean_std(x["bank_mean"], x["bank_std"]) if not math.isnan(x["bank_mean"]) else "--"]
        rows.append(r)
    hdr = ["Variant"] + [f"{escape(labels.get(s, s))}: {k}" for s in scen for k in ("comm.", "bank mean")]
    tex = latex_table(rows, hdr, caption="Ablations (test AP, mean $\\pm$ std over seeds). "
                      "``comm.'' = committee AP (for ensemble rows: the ensemble AP); ``bank mean'' "
                      "= mean AP of the three distilled bank models. The primary method was fixed "
                      "before these runs; ablations are not used to select it.",
                      label="tab:ablations", align="l" + "r" * (2 * len(scen)), size=r"\scriptsize")
    write_table(tdir / "table8_ablations", tex)
    return summ


# --------------------------------------------------------------------------- statistics
def bootstrap_stats(prim: pd.DataFrame, cfg: dict | None, rdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bcfg = ((cfg or {}).get("evaluation", {}) or {}).get("bootstrap", {}) or {}
    B = int(bcfg.get("n_resamples", 1000))
    conf = float(bcfg.get("confidence", 0.95))
    bseed = int(bcfg.get("seed", 0))
    ci_rows, diff_rows = [], []
    for (seed, sc), g in prim.groupby(["seed", "scenario"]):
        preds = {}
        for _, r in g.iterrows():
            f = Path(r["run_dir"]) / "predictions" / f"{r['system']}.npz"
            if f.is_file():
                preds[r["system"]] = load_npz(f)
        if not preds:
            continue
        y = next(iter(preds.values()))["y_test"]
        for s in ["fedavg_global", "ensemble_equal", "kd_committee", "bank_A", "bank_B", "bank_C",
                  "kd_bank_A", "kd_bank_B", "kd_bank_C", "centralized_xgboost"]:
            if s in preds:
                ci = bootstrap_ap_ci(y, preds[s]["p_test"], B, conf, bseed)
                ci_rows.append({"seed": seed, "scenario": sc, "system": s, **ci})
        for a, b in COMPARISONS:
            if a in preds and b in preds:
                if not np.array_equal(preds[a]["y_test"], preds[b]["y_test"]):
                    raise RuntimeError(f"Test labels differ between {a} and {b}")
                d = paired_bootstrap_ap_diff(y, preds[a]["p_test"], preds[b]["p_test"], B, conf, bseed)
                diff_rows.append({"seed": seed, "scenario": sc, "a": a, "b": b, **d})
    ci, diff = pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)
    write_csv(rdir / "bootstrap_ap_ci.csv", ci)
    write_csv(rdir / "bootstrap_paired_ap_diff.csv", diff)
    return ci, diff


# --------------------------------------------------------------------------- figures data
def heterogeneity_series(prim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc, g in prim.groupby("scenario"):
        def add(key, label, vals):
            v = np.asarray(vals, dtype=float)
            rows.append({"scenario": sc, "series": key, "label": label, "mean": v.mean() if len(v) else np.nan,
                         "std": v.std(ddof=1) if len(v) > 1 else np.nan, "n": len(v)})
        for key, label in [("fedavg_global", "FedAvg (homogeneous MLP)"),
                           ("kd_committee", "Heterogeneous KD committee")]:
            add(key, label, g[g["system"] == key]["test_average_precision"])
        add("ensemble", "Heterogeneous ensemble (no KD)",
            g[g["system"] == "ensemble_equal"]["test_average_precision"])
        add("kd_bank_mean", "KD, mean of bank models",
            g[g["system"].str.match(r"kd_bank_[A-Z]$")].groupby("seed")["test_average_precision"].mean())
        add("local_mean", "Local-only, mean of bank models",
            g[g["system"].str.match(r"bank_[A-Z]$")].groupby("seed")["test_average_precision"].mean())
    return pd.DataFrame(rows)


def curve_data(prim: pd.DataFrame, seed: int, scen: list[str]) -> dict:
    out = {}
    want = [("centralized_xgboost", "Centralized (ref.)", "Centralized XGB (ref.)"),
            ("fedavg_global", "FedAvg", "FedAvg MLP"),
            ("ensemble_equal", "Ensemble (no KD)", "Ensemble (no KD)"),
            ("kd_committee", "KD committee", "KD committee")]
    for sc in scen:
        g = prim[(prim["scenario"] == sc) & (prim["seed"] == seed)]
        lst = []
        for s, fam, lab in want:
            r = g[g["system"] == s]
            if r.empty:
                continue
            f = Path(r["run_dir"].iloc[0]) / "predictions" / f"{s}.npz"
            if f.is_file():
                d = load_npz(f)
                lst.append((lab, fam, d["y_test"], d["p_test"]))
        if lst:
            out[sc] = lst
    return out


# --------------------------------------------------------------------------- main
def generate_report(root: str | Path, cfg: dict | None = None) -> Path:
    root = Path(root).resolve()
    if cfg is None and (root / "config.yaml").is_file():
        import yaml

        cfg = yaml.safe_load((root / "config.yaml").read_text())
    tdir, fdir, rdir = root / "tables", root / "figures", root / "reports"
    for d in (tdir, fdir, rdir):
        d.mkdir(parents=True, exist_ok=True)
    data = collect(root)
    status, mdf, rounds, parts = data["status"], data["metrics"], data["rounds"], data["partitions"]
    write_csv(rdir / "run_status.csv", status)
    if mdf.empty:
        (rdir / "summary.md").write_text("# Report\n\nNo completed runs found; nothing to report.\n")
        log.warning("No completed runs under %s", root)
        return rdir
    prov = provenance_summary(root, status)
    publishable = bool((cfg or {}).get("publishable", False)) and not bool(
        ((cfg or {}).get("data") or {}).get("synthetic", False)) and prov["verified"]
    banner = None if publishable else (
        "DEVELOPMENT RESULTS - NOT FOR PUBLICATION" if not prov["verified"] or not (cfg or {}).get(
            "publishable") else "NOT FOR PUBLICATION")

    all_scen = order_scenarios(cfg, mdf["scenario"].unique())
    prim_scen = [s for s in all_scen if not ((cfg or {}).get("scenarios", {}).get(s) or {}).get("ablation")]
    labels = scenario_labels(cfg, all_scen)
    prim = _primary(mdf)
    write_csv(rdir / "all_metrics.csv", mdf)

    present = [m for m in METRICS if m in prim]
    summ = summarize_across_seeds(prim, ["scenario", "system", "family", "short_label", "method"], present)
    summ = _sort_systems(summ)
    write_csv(rdir / "seed_summary.csv", summ)

    pairs = [paired_seed_differences(prim, a, b, "test_average_precision", ["scenario"],
                                     system_col="system") for a, b in COMPARISONS]
    pairs = pd.concat([p for p in pairs if not p.empty], ignore_index=True) if any(
        not p.empty for p in pairs) else pd.DataFrame()
    write_csv(rdir / "paired_seed_differences.csv", pairs)

    hdf = table1(root, parts, all_scen, labels, tdir)
    tables_main(summ, prim_scen, all_scen, labels, tdir)
    t5 = table5(prim, all_scen, labels, tdir)
    cost = table6(mdf, parts, tdir)
    abl = table_ablations(mdf, prim_scen, labels, tdir)
    ci, diff = bootstrap_stats(prim, cfg, rdir)

    figs = {}
    figs["fig1"] = viz.fig_partitions(parts, all_scen, labels, fdir, banner)
    figs["fig2"] = viz.fig_ap_comparison(summ[summ["system"].isin(SYSTEM_ORDER)], prim_scen, labels,
                                         fdir, banner)
    het = heterogeneity_series(prim)
    write_csv(fdir / "fig3_data.csv", het)
    figs["fig3"] = viz.fig_heterogeneity(het, all_scen, labels, fdir, banner)
    figs["fig4"] = viz.fig_rounds(rounds[rounds["variant"] == "primary"] if not rounds.empty else rounds,
                                  prim_scen, labels, fdir, banner)
    seeds = sorted(prim["seed"].unique())
    cseed = int(((cfg or {}).get("report") or {}).get("bootstrap_seed_for_curves", seeds[0]))
    cseed = cseed if cseed in seeds else int(seeds[0])
    cd = curve_data(prim, cseed, prim_scen)
    figs["fig5"] = viz.fig_curves(cd, prim_scen, labels, fdir, "pr", cseed, banner)
    figs["fig6"] = viz.fig_curves(cd, prim_scen, labels, fdir, "roc", cseed, banner)
    figs["fig7"] = viz.fig_before_after(t5, all_scen, labels, fdir, banner)
    figs["fig8"] = viz.fig_cost(cost, fdir, banner)

    _summary_md(root, rdir, cfg, status, summ, pairs, diff, t5, abl, hdf, figs, publishable,
                prim_scen, all_scen, labels, prov)
    log.info("Report written to %s", rdir)
    return rdir


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _summary_md(root, rdir, cfg, status, summ, pairs, diff, t5, abl, hdf, figs, publishable,
                prim_scen, all_scen, labels, prov) -> None:
    n_done = int((status["status"] == "completed").sum())
    failed = status[status["status"] != "completed"]
    L = [f"# Experiment report: `{root.name}`", ""]
    if not publishable:
        reasons = []
        if not (cfg or {}).get("publishable") or ((cfg or {}).get("data") or {}).get("synthetic"):
            reasons.append("this profile uses a subsample and/or synthetic data and reduced models")
        reasons += prov["problems"]
        L += ["> **NOT FOR PUBLICATION.** " + "; ".join(reasons) + ".", ""]
    env = read_json(root / "environment.json") if (root / "environment.json").is_file() else {}
    split = read_json(root / "split_manifest.json") if (root / "split_manifest.json").is_file() else {}
    L += ["## Provenance", "",
          f"- Dataset SHA-256: `{split.get('dataset', {}).get('sha256')}` "
          f"(matches canonical ULB release: {split.get('dataset', {}).get('matches_known_ulb_release')})",
          f"- Experiment identity: `{prov['experiment_hash']}`; configuration hash: "
          f"`{prov['config_hash']}`",
          f"- Source revision(s) of the completed runs: {prov['sources_md']}",
          f"- Source provenance verified for every run: {prov['verified']}",
          f"- Python {env.get('python')}, flwr {env.get('packages', {}).get('flwr')}, "
          f"xgboost {env.get('packages', {}).get('xgboost')}, torch {env.get('packages', {}).get('torch')}, "
          f"scikit-learn {env.get('packages', {}).get('scikit-learn')}",
          f"- Runs completed: {n_done}; not completed: {len(failed)}", ""]
    if len(failed):
        L += ["### Runs that did not complete (excluded, not imputed)", "",
              _md_table(failed[["path", "status", "error"]].fillna("")), ""]
    L += ["## Test AP by system (mean over seeds; SDs in `tables/table7_seed_summary.csv`)", ""]
    piv = summ.pivot_table(index="short_label", columns="scenario",
                           values="test_average_precision_mean", sort=False)
    piv = piv[[s for s in all_scen if s in piv.columns]]
    L += [_md_table(piv.reset_index().rename(columns=labels)), "",
          "Seeds share one test set, so the spread across seeds reflects training variability "
          "(partition + initialisation) only, not test-sample uncertainty. Test-sample "
          "uncertainty is in `reports/bootstrap_*.csv` (class-stratified bootstrap).", ""]
    if not pairs.empty:
        L += ["## Seed-paired AP differences (a − b), descriptive", ""]
        L += [_md_table(pairs[["scenario", "a", "b", "n_pairs", "mean_diff", "std_diff",
                               "min_diff", "max_diff", "n_a_better"]]), ""]
        L += ["Statements generated from the table above (no significance testing):", ""]
        for _, r in pairs.iterrows():
            verb = "higher" if r["mean_diff"] > 0 else "lower"
            L.append(f"- {labels.get(r['scenario'], r['scenario'])}: `{r['a']}` AP is on average "
                     f"{abs(r['mean_diff']):.4f} {verb} than `{r['b']}` "
                     f"(range {r['min_diff']:+.4f} to {r['max_diff']:+.4f}; `{r['a']}` higher in "
                     f"{r['n_a_better']}/{r['n_pairs']} seeds).")
        L.append("")
    if not diff.empty:
        agg = diff.assign(ci_excludes_zero=(diff["ci_low"] > 0) | (diff["ci_high"] < 0))
        L += ["## Paired bootstrap of test AP differences (per seed)", "",
              _md_table(agg[["seed", "scenario", "a", "b", "diff", "ci_low", "ci_high",
                             "ci_excludes_zero"]]), "",
              "Each interval reflects test-example resampling for one trained pair of models; "
              "intervals from different seeds are not independent replications. "
              "`ci_excludes_zero` is descriptive: it is not a significance test, no "
              "multiple-comparison correction is applied, and it says nothing about "
              "variation across training seeds or partitions.", ""]
    if not abl.empty:
        L += ["## Ablations (test AP; committee and mean bank model)", "", _md_table(abl), ""]
    L += ["## Figures", ""]
    for k, v in figs.items():
        L.append(f"- {k}: " + (", ".join(Path(p).name for p in v) if v else "not generated (no data)"))
    L += ["", "## Tables", ""] + [f"- `{p.name}`" for p in sorted((root / "tables").glob("*.tex"))]
    (rdir / "summary.md").write_text("\n".join(L) + "\n")
