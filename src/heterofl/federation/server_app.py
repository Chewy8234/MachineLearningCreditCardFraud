"""Flower ServerApp: dispatches a federated job to FedAvg or heterogeneous KD.

Two entry points share ``run_job``:

* ``app`` -- reads ``job-path`` from the Flower run config; used by
  ``flwr run . --run-config 'job-path="..."'`` (development/debugging).
* ``make_server_app(job_path)`` -- used by ``heterofl.federation.simulation``,
  which the experiment runner launches through the Flower Simulation Runtime.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from heterofl.federation.fedavg import run_fedavg
from heterofl.federation.heterogeneous_strategy import HeterogeneousKDStrategy, identify_clients
from heterofl.federation.job import load_job
from heterofl.utils.io import write_json
from heterofl.utils.logging import FileLog, get_logger, setup_logging

log = get_logger("federation.server")


def run_job(grid: Grid, job_path: str) -> dict:
    job = load_job(str(job_path))
    server_dir = Path(job.job_dir) / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    status_path = server_dir / "status.json"
    write_json(status_path, {"status": "running", "started": time.time()})
    setup_logging()
    with FileLog(server_dir / "server.log", logging.INFO):
        try:
            mapping = identify_clients(grid, job)
            if job.mode == "fedavg":
                summary = run_fedavg(grid, job, {n: s.bank for n, s in mapping.items()})
            else:
                summary = HeterogeneousKDStrategy(job).run(grid, mapping)
        except BaseException as exc:
            write_json(status_path, {"status": "failed", "error": repr(exc),
                                     "traceback": traceback.format_exc()})
            log.error("Federated job failed: %r", exc)
            raise
    write_json(status_path, {"status": "completed", "finished": time.time(),
                             "node_to_bank": {str(n): s.bank for n, s in mapping.items()}})
    return summary


def make_server_app(job_path: str) -> ServerApp:
    server = ServerApp()

    @server.main()
    def _main(grid: Grid, context: Context) -> None:
        run_job(grid, job_path)

    return server


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    job_path = str(context.run_config.get("job-path", ""))
    if not job_path:
        raise ValueError(
            "No job-path in the run config. Create a job with\n"
            "  python -m heterofl.experiments.runner --config configs/smoke.yaml "
            "--prepare-job kd\nand pass the printed path via --run-config.")
    run_job(grid, job_path)
