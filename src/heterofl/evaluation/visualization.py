"""Publication figures (PNG + PDF), drawn only from recorded results.

Every function receives already-aggregated data and returns the list of files
written, or ``[]`` when there is nothing to draw -- figures are never produced
from empty or placeholder data.

Colour encodes the *method family* (fixed slot order, never cycled) or the
*bank*; it never encodes rank. The categorical palette is a validated
CVD-safe order (adjacent-pair checks); three slots fall below 3:1 contrast on
white, so bars carry value labels and lines carry distinct markers/linestyles,
and every figure has a CSV table alongside it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAMILY_ORDER = ["Local", "Centralized (ref.)", "Ensemble (no KD)", "FedAvg", "KD (per bank)",
                "KD committee"]
FAMILY_COLORS = dict(zip(FAMILY_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                        "#e87ba4", "#008300"], strict=True))
BANK_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
MODEL_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest", "mlp": "MLP"}
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _save(fig, out: Path, name: str, banner: str | None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    if banner:
        # below all content (incl. bottom legends); bbox_inches="tight" keeps it in frame
        fig.text(0.5, -0.09, banner, ha="center", va="top", fontsize=7, color="#d03b3b")
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _bar_labels(ax, bars, values, fmt="{:.3f}", horizontal=False):
    for b, v in zip(bars, values, strict=True):
        if v is None or np.isnan(v):
            continue
        if horizontal:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                    va="center", ha="left", fontsize=7, color=TEXT)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=6.5, color=TEXT)


def fig_partitions(parts: pd.DataFrame, scenarios: list[str], labels: dict, out: Path,
                   banner=None) -> list[Path]:
    """Figure 1: fraud rate and data share of each bank, per scenario and seed."""
    if parts.empty:
        return []
    seeds = sorted(parts["partition_seed"].unique())
    fig, axes = plt.subplots(2, len(scenarios), figsize=(2.6 * len(scenarios) + 0.6, 4.6),
                             squeeze=False, sharey="row")
    w = 0.26
    for j, sc in enumerate(scenarios):
        d = parts[parts["scenario"] == sc]
        for row, (col, ylab, fmt) in enumerate([("fraud_pct", "Fraud rate (%)", "{:.2f}"),
                                                ("relative_size", "Share of client data", "{:.2f}")]):
            ax = axes[row][j]
            x = np.arange(len(seeds))
            for k, bank in enumerate(sorted(d["bank"].unique())):
                vals = [float(d[(d["partition_seed"] == s) & (d["bank"] == bank)][col].iloc[0])
                        for s in seeds]
                bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=BANK_COLORS.get(bank),
                              label=f"Bank {bank}", edgecolor="white", linewidth=0.6)
                if len(seeds) <= 3:
                    _bar_labels(ax, bars, vals, fmt)
            ax.set_xticks(x, [f"seed {s}" for s in seeds])
            if row == 0:
                ax.set_title(labels.get(sc, sc))
            if j == 0:
                ax.set_ylabel(ylab)
            ax.grid(axis="x", visible=False)
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.suptitle("Client class distributions as realised (not implied by α)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig1_fraud_distribution", banner)


def fig_ap_comparison(summary: pd.DataFrame, scenarios: list[str], labels: dict, out: Path,
                      banner=None) -> list[Path]:
    """Figure 2: test AP (mean ± std over seeds) of every system, per scenario."""
    if summary.empty:
        return []
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.3 * len(scenarios), 4.4),
                             squeeze=False, sharex=True)
    for j, sc in enumerate(scenarios):
        ax = axes[0][j]
        d = summary[summary["scenario"] == sc].iloc[::-1]
        if d.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(d))
        bars = ax.barh(y, d["test_average_precision_mean"], 0.72,
                       xerr=d["test_average_precision_std"].fillna(0),
                       color=[FAMILY_COLORS[f] for f in d["family"]],
                       error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": TEXT_2},
                       edgecolor="white", linewidth=0.6)
        means = d["test_average_precision_mean"].to_numpy()
        ends = means + d["test_average_precision_std"].fillna(0).to_numpy()
        for yi, m, e in zip(y, means, ends, strict=True):  # label past the error bar
            ax.text(e + 0.012, yi, f"{m:.3f}", va="center", ha="left", fontsize=7, color=TEXT)
        ax.set_yticks(y, d["short_label"])
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Test Average Precision (AP)")
        ax.set_title(labels.get(sc, sc))
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_ORDER
               if f in set(summary["family"])]
    fig.legend(handles, [f for f in FAMILY_ORDER if f in set(summary["family"])],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig2_ap_comparison", banner)


def fig_heterogeneity(summary: pd.DataFrame, scenarios: list[str], labels: dict, out: Path,
                      banner=None) -> list[Path]:
    """Figure 3: AP of FedAvg vs heterogeneous KD (and controls) across heterogeneity levels."""
    if summary.empty or len(scenarios) < 2:
        return []
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(scenarios))
    for i, (key, fam) in enumerate([("fedavg_global", "FedAvg"),
                                    ("kd_committee", "KD committee"),
                                    ("ensemble", "Ensemble (no KD)"),
                                    ("kd_bank_mean", "KD (per bank)"),
                                    ("local_mean", "Local")]):
        d = summary[summary["series"] == key].set_index("scenario").reindex(scenarios)
        if d["mean"].isna().all():
            continue
        ax.errorbar(x, d["mean"], yerr=d["std"].fillna(0), color=FAMILY_COLORS[fam],
                    marker=MARKERS[i], linestyle=LINESTYLES[i], linewidth=1.6, markersize=6,
                    capsize=3, label=d["label"].dropna().iloc[0] if d["label"].notna().any() else key)
    ax.set_xticks(x, [labels.get(s, s) for s in scenarios])
    ax.set_ylabel("Test AP (mean ± std over seeds)")
    ax.set_ylim(0, 1)
    ax.set_title("Detection performance vs. client heterogeneity")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    return _save(fig, out, "fig3_heterogeneity", banner)


def fig_rounds(rounds: pd.DataFrame, scenarios: list[str], labels: dict, out: Path,
               banner=None) -> list[Path]:
    """Figure 4: validation AP per round (mean ± std over seeds)."""
    if rounds.empty:
        return []
    fig, axes = plt.subplots(2, len(scenarios), figsize=(2.9 * len(scenarios), 5.0),
                             squeeze=False, sharey=True)
    for j, sc in enumerate(scenarios):
        d = rounds[rounds["scenario"] == sc]
        top, bot = axes[0][j], axes[1][j]
        for i, (sysname, fam, lab) in enumerate([("fedavg_global", "FedAvg", "FedAvg global MLP"),
                                                 ("kd_committee", "KD committee", "KD committee")]):
            g = d[d["system"] == sysname].groupby("round")["val_average_precision"].agg(["mean", "std"])
            g = g[g.index >= (1 if sysname == "fedavg_global" else 0)]
            if g.empty:
                continue
            top.plot(g.index, g["mean"], color=FAMILY_COLORS[fam], marker=MARKERS[i],
                     linestyle=LINESTYLES[i], linewidth=1.6, markersize=5, label=lab)
            top.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                             color=FAMILY_COLORS[fam], alpha=0.15, linewidth=0)
        for i, bank in enumerate(sorted(BANK_COLORS)):
            g = d[d["system"] == f"kd_bank_{bank}"]
            if g.empty:
                continue
            mt = MODEL_NAMES.get(g["model_type"].iloc[0], g["model_type"].iloc[0])
            g = g.groupby("round")["val_average_precision"].agg(["mean", "std"])
            bot.plot(g.index, g["mean"], color=BANK_COLORS[bank], marker=MARKERS[i],
                     linestyle=LINESTYLES[i], linewidth=1.6, markersize=5,
                     label=f"Bank {bank} ({mt})")
            bot.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                             color=BANK_COLORS[bank], alpha=0.15, linewidth=0)
        top.set_title(labels.get(sc, sc))
        bot.set_xlabel("Round (KD round 0 = before distillation)")
        for ax in (top, bot):
            ax.set_ylim(0, 1)
            ax.xaxis.get_major_locator().set_params(integer=True)
        if j == 0:
            top.set_ylabel("Validation AP")
            bot.set_ylabel("Validation AP (KD per bank)")
    axes[0][0].legend(frameon=False, loc="lower right")
    axes[1][0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, out, "fig4_validation_rounds", banner)


def fig_curves(curves: dict, scenarios: list[str], labels: dict, out: Path, kind: str,
               seed: int, banner=None) -> list[Path]:
    """Figures 5/6: PR or ROC curves of the final models (one seed, test split)."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    if not curves:
        return []
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.0 * len(scenarios), 3.2),
                             squeeze=False)
    for j, sc in enumerate(scenarios):
        ax = axes[0][j]
        for i, (lab, fam, y, p) in enumerate(curves.get(sc, [])):
            if kind == "pr":
                prec, rec, _ = precision_recall_curve(y, p)
                ax.plot(rec, prec, color=FAMILY_COLORS[fam], linestyle=LINESTYLES[i],
                        linewidth=1.5, label=lab, drawstyle="steps-post")
            else:
                fpr, tpr, _ = roc_curve(y, p)
                # empirical ROC is a step function; steps avoid fake diagonals on a log axis
                ax.plot(np.maximum(fpr, 1e-5), tpr, color=FAMILY_COLORS[fam],
                        linestyle=LINESTYLES[i], linewidth=1.5, label=lab, drawstyle="steps-post")
        if kind == "pr":
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision" if j == 0 else "")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
        else:
            ax.set_xscale("log")
            ax.set_xlim(1e-5, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("False positive rate (log scale)")
            ax.set_ylabel("True positive rate" if j == 0 else "")
        ax.set_title(labels.get(sc, sc))
    axes[0][0].legend(frameon=False, fontsize=7, loc="lower left" if kind == "pr" else "upper left")
    name = "fig5_pr_curves" if kind == "pr" else "fig6_roc_curves"
    fig.suptitle(f"{'Precision-recall' if kind == 'pr' else 'ROC'} curves on the test split "
                 f"(seed {seed})", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, name, banner)


def fig_before_after(df: pd.DataFrame, scenarios: list[str], labels: dict, out: Path,
                     banner=None) -> list[Path]:
    """Figure 7: per-bank test AP before (local-only) and after KD."""
    if df.empty:
        return []
    fig, axes = plt.subplots(1, len(scenarios), figsize=(2.8 * len(scenarios), 3.3),
                             squeeze=False, sharey=True)
    w = 0.38
    for j, sc in enumerate(scenarios):
        ax = axes[0][j]
        d = df[df["scenario"] == sc].sort_values("bank")
        x = np.arange(len(d))
        b1 = ax.bar(x - w / 2, d["before_mean"], w * 0.92, yerr=d["before_std"].fillna(0),
                    color="white", edgecolor=[BANK_COLORS[b] for b in d["bank"]], linewidth=1.4,
                    hatch="///", error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": TEXT_2})
        b2 = ax.bar(x + w / 2, d["after_mean"], w * 0.92, yerr=d["after_std"].fillna(0),
                    color=[BANK_COLORS[b] for b in d["bank"]], edgecolor="white",
                    error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": TEXT_2})
        _bar_labels(ax, b1, d["before_mean"].tolist())
        _bar_labels(ax, b2, d["after_mean"].tolist())
        ax.set_xticks(x, [f"Bank {b}\n{MODEL_NAMES.get(m, m)}"
                          for b, m in zip(d["bank"], d["model_type"], strict=True)])
        ax.set_ylim(0, 1.08)
        ax.set_title(labels.get(sc, sc))
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.set_ylabel("Test AP (mean ± std over seeds)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=TEXT_2, hatch="///"),
               plt.Rectangle((0, 0), 1, 1, color=TEXT_2)]
    fig.legend(handles, ["Before: local-only model", "After: heterogeneous KD"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "fig7_before_after_kd", banner)


def fig_cost(cost: pd.DataFrame, out: Path, banner=None) -> list[Path]:
    """Figure 8: communication volume and training time (two separate panels; no dual axis)."""
    if cost.empty:
        return []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    d = cost.copy()
    y = np.arange(len(d))[::-1]
    col = [FAMILY_COLORS[f] for f in d["family"]]
    comm_raw = d["communication_mb_mean"].to_numpy(dtype=float)
    comm = np.nan_to_num(comm_raw)
    b = a1.barh(y, comm, 0.7, xerr=d["communication_mb_std"].fillna(0), color=col,
                error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": TEXT_2})
    _bar_labels(a1, b, comm_raw.tolist(), fmt="{:.2f}", horizontal=True)
    for yi, v in zip(y, comm_raw, strict=True):
        if np.isnan(v):  # centralized: not a protocol -- pooling raw data instead
            a1.text(0, yi, " n/a (pools raw data)", va="center", ha="left", fontsize=7, color=TEXT_2)
    a1.set_yticks(y, d["label"])
    a1.set_xlabel("Communication per run (MB, up + down)")
    a1.set_xlim(0, max(np.nanmax(comm) * 1.3, 0.1))
    t = d["train_seconds_mean"].to_numpy()
    b = a2.barh(y, t, 0.7, xerr=d["train_seconds_std"].fillna(0), color=col,
                error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": TEXT_2})
    _bar_labels(a2, b, t.tolist(), fmt="{:.1f}", horizontal=True)
    a2.set_yticks(y, [""] * len(y))
    a2.set_xlabel("Total client training time (s)")
    a2.set_xlim(0, max(np.nanmax(t) * 1.3, 1))
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    fig.suptitle("Overhead per run (mean ± std over seeds and scenarios)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig8_cost", banner)
