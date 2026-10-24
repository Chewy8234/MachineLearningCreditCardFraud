"""Shared harness for all experimental conditions.

Every condition produces "systems" (a trained model or a fixed combination of
models). Each system is evaluated identically: threshold chosen on validation
with the configured rule, then all metrics computed on the test split. The
harness is the only code that reads test data, and only after training and
model selection for that condition are finished.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from heterofl.data.pipeline import ScenarioData
from heterofl.evaluation.metrics import evaluate_system, flatten_evaluation
from heterofl.models.base import FraudModel
from heterofl.utils.io import save_npz, write_csv, write_json
from heterofl.utils.logging import get_logger

log = get_logger("experiments")

MODEL_LABELS = {"xgboost": "XGBoost", "random_forest": "Random Forest", "mlp": "MLP"}


@dataclass
class RunContext:
    cfg: dict
    seed: int
    scenario: str
    method: str
    data: ScenarioData
    out_dir: Path
    run_id: str
    variant: str = "primary"            # "primary" or the ablation name
    #: run identity (``heterofl.experiments.identity.run_identity``); None for debug jobs
    identity: dict | None = None
    rows: list[dict] = field(default_factory=list)
    evaluations: dict[str, dict] = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)

    @property
    def clients(self) -> list[dict]:
        return self.cfg["clients"]

    @property
    def n_jobs(self) -> int:
        return int(self.cfg["federation"]["client_num_cpus"])

    @property
    def models_cfg(self) -> dict:
        return self.cfg["models"]

    def results_root(self) -> Path:
        """``results/<profile>/seed_<s>/<scenario>`` for this run."""
        p = self.out_dir.parent
        return p.parent if p.name == "ablations" else p

    def eval_cfg(self) -> tuple[str, tuple[float, ...]]:
        ev = self.cfg["evaluation"]
        return ev.get("threshold_rule", "max_f1"), tuple(ev.get("fixed_fpr_targets", (0.001, 0.01)))


def timed_predict(model: FraudModel, X: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = model.predict_proba(X)
    return p, time.perf_counter() - t0


def record_system(ctx: RunContext, *, system: str, label: str, model_type: str,
                  p_val: np.ndarray, p_test: np.ndarray, extra: dict | None = None) -> dict:
    """Evaluate one system (threshold from validation) and store predictions + metrics."""
    y_val = ctx.data.validation_labels()
    y_test = ctx.data.test_labels()
    rule, fprs = ctx.eval_cfg()
    ev = evaluate_system(y_val, p_val, y_test, p_test, rule=rule, fpr_targets=fprs)
    save_npz(ctx.out_dir / "predictions" / f"{system}.npz", p_val=np.asarray(p_val),
             p_test=np.asarray(p_test), y_val=y_val, y_test=y_test)
    ident = ctx.identity or {}
    row = {"run_id": ctx.run_id, "profile": ctx.cfg["profile"], "seed": ctx.seed,
           "scenario": ctx.scenario, "method": ctx.method, "variant": ctx.variant,
           "config_hash": ident.get("config_hash"),
           "partition_sha256": ident.get("partition_sha256"),
           "system": system, "system_label": label, "model_type": model_type,
           **(extra or {}), **flatten_evaluation(ev)}
    ctx.rows.append(row)
    ctx.evaluations[system] = {"label": label, "model_type": model_type, **(extra or {}), **ev}
    t = ev["test"]
    log.info("[%s/%s/seed %d] %-28s test AP=%.4f ROC-AUC=%.4f F1@val-thr=%.4f",
             ctx.scenario, ctx.method, ctx.seed, system, t["average_precision"], t["roc_auc"],
             t["at_val_threshold"]["f1"])
    return row


def write_outputs(ctx: RunContext, extra: dict | None = None) -> None:
    if not ctx.rows:
        raise RuntimeError(f"{ctx.method}: no systems were evaluated; refusing to write "
                           "empty metrics")
    write_csv(ctx.out_dir / "metrics.csv", ctx.rows)
    write_json(ctx.out_dir / "metrics.json", {
        "run_id": ctx.run_id, "seed": ctx.seed, "scenario": ctx.scenario,
        "method": ctx.method, "variant": ctx.variant,
        "publishable": bool(ctx.cfg.get("publishable", False)),
        "split_note": "validation = threshold/model selection; test = final evaluation only",
        "systems": ctx.evaluations, **(extra or {})})
    write_json(ctx.out_dir / "runtime.json", ctx.runtime)
