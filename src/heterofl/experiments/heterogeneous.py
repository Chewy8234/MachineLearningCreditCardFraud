"""Experiment 5 -- proposed heterogeneous federated knowledge distillation.

Runs the Flower KD federation (``HeterogeneousKDStrategy``), then evaluates:

* each bank's distilled model (``kd_bank_<X>``): what each institution would
  deploy after collaborating, and
* the committee of distilled models (``kd_committee``), combined exactly like
  the non-distilled ensemble (same weighting rule), so RQ3 compares
  committee-with-KD against committee-without-KD on equal terms.

Round-0 models (before any distillation) are evaluated as well
(``kd_bank_<X>_round0``), giving a within-run before/after comparison. They
should coincide with the local-only models; the measured difference is stored
as a determinism diagnostic.
"""

from __future__ import annotations

import numpy as np

from heterofl.distillation.aggregation import weighted_average
from heterofl.evaluation.metrics import average_precision, roc_auc, threshold_metrics
from heterofl.experiments.common import MODEL_LABELS, RunContext, record_system, timed_predict
from heterofl.experiments.ensemble import committee_weights
from heterofl.experiments.federated_common import build_job, job_dir, launch, select_round
from heterofl.models.model_factory import load_model, model_seed
from heterofl.utils.io import load_npz, read_json, write_csv, write_json


def _round_dir(jd, bank: str, rnd: int):
    return jd / "clients" / f"bank_{bank}" / f"round_{rnd:02d}"


def run(ctx: RunContext) -> None:
    job = build_job(ctx, "kd")
    launch(ctx, job)
    jd = job_dir(ctx)
    summary = read_json(jd / "server" / "rounds.json")
    banks = [c["bank"] for c in ctx.clients]
    mtypes = {c["bank"]: c["model"] for c in ctx.clients}
    rounds = [int(r["round"]) for r in summary["rounds"]]
    weighting = ctx.cfg["ensemble"].get("weighting", "equal")

    Xv, y_val = ctx.data.validation()
    Xt, _ = ctx.data.test()

    # ---- validation curves (per bank and committee), from saved client outputs
    val_probs = {r: {b: np.load(_round_dir(jd, b, r) / "val_probs.npy") for b in banks}
                 for r in rounds}
    curves: dict[str, dict[int, float]] = {f"kd_bank_{b}": {} for b in banks}
    curves["kd_committee"] = {}
    round_rows = []
    by_round = {int(r["round"]): r for r in summary["rounds"]}
    for r in rounds:
        w = committee_weights(val_probs[r], y_val, weighting)
        com = weighted_average(val_probs[r], sorted(banks), w)
        entries = {**{f"kd_bank_{b}": val_probs[r][b] for b in banks}, "kd_committee": com}
        for system, p in entries.items():
            ap = average_precision(y_val, p)
            curves[system][r] = ap
            tm = threshold_metrics(y_val, p, 0.5)
            bank = system.removeprefix("kd_bank_") if system != "kd_committee" else None
            cm = (by_round[r]["client_metrics"].get(bank, {}) if bank else {})
            round_rows.append({
                "seed": ctx.seed, "scenario": ctx.scenario, "method": ctx.method,
                "variant": ctx.variant, "round": r, "system": system,
                "model_type": mtypes.get(bank, "committee"),
                "val_average_precision": ap, "val_roc_auc": roc_auc(y_val, p),
                "val_precision@0.5": tm["precision"], "val_recall@0.5": tm["recall"],
                "val_f1@0.5": tm["f1"],
                "bytes_up": by_round[r]["bytes_up"], "bytes_down": by_round[r]["bytes_down"],
                "round_seconds": by_round[r]["round_seconds"],
                "fit_seconds": cm.get("fit-seconds"),
                "client_reported_val_ap": cm.get("val-ap"),
                "fit_total_proxy_weight": cm.get("fit-total_proxy_weight"),
                "fit_total_local_weight": cm.get("fit-total_local_weight"),
                "fit_final_distillation_loss": cm.get("fit-final_distillation_loss"),
            })
    write_csv(ctx.out_dir / "rounds.csv", round_rows)

    # ---- model selection (validation only), then test evaluation
    policy = ctx.cfg["evaluation"].get("model_selection", "final_round")
    candidates = [r for r in rounds if r >= 1] or rounds
    comm = summary["total_bytes_up"] + summary["total_bytes_down"]
    fit_total = sum(float(m.get("fit-seconds", 0.0)) for r in summary["rounds"]
                    for m in r["client_metrics"].values())
    test_probs, val_sel, chosen = {}, {}, {}
    infer_total = 0.0
    for b in banks:
        r_sel = select_round(curves[f"kd_bank_{b}"], policy, candidates)
        chosen[b] = r_sel
        model = _load(ctx, jd, b, mtypes[b], r_sel)
        val_sel[b], _ = timed_predict(model, Xv)
        test_probs[b], infer_s = timed_predict(model, Xt)
        infer_total += infer_s
        if not np.allclose(val_sel[b], val_probs[r_sel][b], atol=1e-6):
            raise RuntimeError(f"Reloaded bank {b} round {r_sel} model does not reproduce its "
                               "saved validation predictions")
        bank_fit = sum(float(r["client_metrics"][b].get("fit-seconds", 0.0))
                       for r in summary["rounds"])
        record_system(ctx, system=f"kd_bank_{b}",
                      label=f"KD {MODEL_LABELS[mtypes[b]]} (Bank {b})", model_type=mtypes[b],
                      p_val=val_sel[b], p_test=test_probs[b],
                      extra={"bank": b, "kd_phase": "after", "selected_round": r_sel,
                             "model_selection": policy, "train_seconds": bank_fit,
                             "test_inference_seconds": infer_s,
                             "communication_bytes": comm / len(banks)})

    # committee: one round for all members (chosen on committee validation AP)
    r_com = select_round(curves["kd_committee"], policy, candidates)
    com_val, com_test = {}, {}
    for b in banks:
        if chosen[b] == r_com:
            com_val[b], com_test[b] = val_sel[b], test_probs[b]
        else:
            m = _load(ctx, jd, b, mtypes[b], r_com)
            com_val[b], com_test[b] = m.predict_proba(Xv), m.predict_proba(Xt)
    w = committee_weights(com_val, y_val, weighting)
    record_system(ctx, system="kd_committee",
                  label=f"Heterogeneous KD committee ({weighting})", model_type="committee",
                  p_val=weighted_average(com_val, sorted(banks), w),
                  p_test=weighted_average(com_test, sorted(banks), w),
                  extra={"kd_phase": "after", "selected_round": r_com, "model_selection": policy,
                         "weighting": weighting, "train_seconds": fit_total,
                         "test_inference_seconds": infer_total,
                         "federation_wall_seconds": summary["wall_seconds"],
                         "communication_bytes": comm})

    # ---- round-0 (pre-distillation) models, and determinism vs. local-only
    diag = {}
    local_dir = ctx.results_root() / "local"
    for b in banks:
        m0 = _load(ctx, jd, b, mtypes[b], 0)
        p0_test, _ = timed_predict(m0, Xt)
        record_system(ctx, system=f"kd_bank_{b}_round0",
                      label=f"KD round 0 {MODEL_LABELS[mtypes[b]]} (Bank {b})",
                      model_type=mtypes[b], p_val=val_probs[0][b], p_test=p0_test,
                      extra={"bank": b, "kd_phase": "before", "selected_round": 0})
        lp = local_dir / "predictions" / f"bank_{b}.npz"
        if lp.is_file():
            lv = load_npz(lp)["p_val"]
            diag[b] = {"max_abs_diff_val_probs_vs_local": float(np.max(np.abs(lv - val_probs[0][b]))),
                       "val_ap_local": average_precision(y_val, lv),
                       "val_ap_round0": average_precision(y_val, val_probs[0][b])}
    write_json(ctx.out_dir / "diagnostics.json", {
        "round0_vs_local_only": diag,
        "note": "Round-0 KD models use the same data, seeds and hyperparameters as the "
                "local-only models; non-zero differences indicate nondeterminism "
                "(e.g. thread scheduling), not a protocol difference.",
        "selected_rounds": {**{f"kd_bank_{b}": chosen[b] for b in banks}, "kd_committee": r_com}})
    ctx.runtime["heterogeneous_kd"] = {
        "federation_wall_seconds": summary["wall_seconds"], "client_fit_seconds_total": fit_total,
        "test_inference_seconds": infer_total, "bytes_up": summary["total_bytes_up"],
        "bytes_down": summary["total_bytes_down"], "n_proxy": summary["n_proxy"]}


def _load(ctx: RunContext, jd, bank: str, mtype: str, rnd: int):
    meta = read_json(_round_dir(jd, bank, rnd) / "meta.json")
    return load_model(mtype, _round_dir(jd, bank, rnd) / meta["model_file"], ctx.models_cfg,
                      model_seed(ctx.seed, f"bank_{bank}", mtype), n_jobs=ctx.n_jobs)
