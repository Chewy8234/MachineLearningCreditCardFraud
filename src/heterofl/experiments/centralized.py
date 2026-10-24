"""Experiment 2 -- centralized reference (NOT a federated method).

Each model family is trained on the pooled client-training partitions (the
same rows the banks hold, nothing more: no proxy, validation or test rows).
This is a hypothetical reference for what pooling would achieve under this
simulation; it is not assumed to be an upper bound.
"""

from __future__ import annotations

from heterofl.distillation.trainers import initial_fit
from heterofl.experiments.common import MODEL_LABELS, RunContext, record_system, timed_predict
from heterofl.models.model_factory import create_model, model_seed


def run(ctx: RunContext) -> None:
    X, y = ctx.data.pooled_clients()
    Xv, _ = ctx.data.validation()
    Xt, _ = ctx.data.test()
    model_types = list(dict.fromkeys(c["model"] for c in ctx.clients))
    for mtype in model_types:
        model = create_model(mtype, ctx.models_cfg, model_seed(ctx.seed, "centralized", mtype),
                             n_jobs=ctx.n_jobs, n_features=X.shape[1])
        info = initial_fit(model, X, y)
        model.save(ctx.out_dir / "models" / f"centralized_{mtype}")
        p_val, _ = timed_predict(model, Xv)
        p_test, infer_s = timed_predict(model, Xt)
        ctx.runtime[f"centralized_{mtype}"] = {"train_seconds": info["fit_seconds"],
                                               "test_inference_seconds": infer_s,
                                               "n_train": int(len(y))}
        record_system(ctx, system=f"centralized_{mtype}",
                      label=f"Centralized {MODEL_LABELS[mtype]} (reference)", model_type=mtype,
                      p_val=p_val, p_test=p_test,
                      extra={"train_seconds": info["fit_seconds"],
                             "test_inference_seconds": infer_s, "communication_bytes": None})
