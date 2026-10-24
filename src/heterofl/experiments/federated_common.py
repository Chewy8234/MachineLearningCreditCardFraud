"""Building and launching federated jobs, and round selection from validation only."""

from __future__ import annotations

import math
from pathlib import Path

from heterofl.experiments.common import RunContext
from heterofl.federation.job import ClientSpec, FederatedJob
from heterofl.federation.simulation import launch_subprocess


def build_job(ctx: RunContext, mode: str) -> FederatedJob:
    fed = ctx.cfg["federation"]
    clients = [ClientSpec(partition_id=i, bank=c["bank"], model_type=c["model"])
               for i, c in enumerate(ctx.clients)]
    job = FederatedJob(
        mode=mode,
        job_dir=str((ctx.out_dir / "federation").resolve()),
        view_dir=str(ctx.data.view_dir.resolve()),
        seed=int(ctx.seed),
        num_rounds=int(fed["num_rounds"]),
        clients=clients,
        models=ctx.models_cfg,
        fedavg=ctx.cfg["fedavg"],
        kd=ctx.cfg["kd"],
        expected_proxy_sha256=ctx.data.manifest["proxy_row_id_sha256"],
        num_cpus_per_client=int(fed["client_num_cpus"]),
        num_gpus_per_client=float(fed.get("client_num_gpus", 0.0)),
        round_timeout=float(fed.get("round_timeout", 3600)),
        debug=dict(ctx.cfg.get("debug", {}) or {}),
    )
    job.save()
    return job


def launch(ctx: RunContext, job: FederatedJob) -> None:
    timeout = ctx.cfg["federation"].get("job_timeout")
    launch_subprocess(job.path, ctx.out_dir / "federation.log",
                      timeout=float(timeout) if timeout else None)


def select_round(val_ap_by_round: dict[int, float], policy: str, candidates: list[int]) -> int:
    """Pick the round to evaluate on test, using VALIDATION AP only.

    ``final_round`` (default, fixed a priori) ignores the curve entirely;
    ``best_validation_ap`` takes the argmax (earliest round on ties).
    """
    if policy == "final_round":
        return max(candidates)
    if policy == "best_validation_ap":
        scored = [(r, val_ap_by_round[r]) for r in candidates
                  if r in val_ap_by_round and not math.isnan(val_ap_by_round[r])]
        if not scored:
            return max(candidates)
        best = max(v for _, v in scored)
        return min(r for r, v in scored if v == best)
    raise ValueError(f"Unknown model selection policy {policy!r}")


def job_dir(ctx: RunContext) -> Path:
    return ctx.out_dir / "federation"
