"""Experiment 1 -- local-only models.

Each bank trains its assigned model on its own private partition. No federation,
no knowledge exchange, no cross-bank data. The trained models are also the
members of the non-distilled heterogeneous ensemble (Experiment 4), and they are
trained with the same seeds as the round-0 models of the KD federation.
"""

from __future__ import annotations

from heterofl.distillation.trainers import initial_fit
from heterofl.experiments.common import MODEL_LABELS, RunContext, record_system, timed_predict
from heterofl.models.model_factory import create_model, model_seed


def run(ctx: RunContext) -> None:
    Xv, _ = ctx.data.validation()
    Xt, _ = ctx.data.test()
    for client in ctx.clients:
        bank, mtype = client["bank"], client["model"]
        X, y = ctx.data.bank(bank)
        model = create_model(mtype, ctx.models_cfg, model_seed(ctx.seed, f"bank_{bank}", mtype),
                             n_jobs=ctx.n_jobs, n_features=X.shape[1])
        info = initial_fit(model, X, y)
        model.save(ctx.out_dir / "models" / f"bank_{bank}")
        p_val, _ = timed_predict(model, Xv)
        p_test, infer_s = timed_predict(model, Xt)
        ctx.runtime[f"bank_{bank}"] = {"train_seconds": info["fit_seconds"],
                                       "test_inference_seconds": infer_s,
                                       "n_train": int(len(y)), "n_train_fraud": int(y.sum())}
        record_system(ctx, system=f"bank_{bank}",
                      label=f"Local {MODEL_LABELS[mtype]} (Bank {bank})", model_type=mtype,
                      p_val=p_val, p_test=p_test,
                      extra={"bank": bank, "train_seconds": info["fit_seconds"],
                             "test_inference_seconds": infer_s, "communication_bytes": 0})
