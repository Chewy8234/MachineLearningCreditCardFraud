"""Launch one federated job through the Flower Simulation Runtime.

    python -m heterofl.federation.simulation --job <job_dir>/job.json

The experiment harness runs this module in a subprocess per job, which
isolates Ray from the parent process, captures the complete log, and makes a
crashed job a recorded failure instead of a crashed runner.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from heterofl.utils.io import read_json
from heterofl.utils.logging import get_logger

log = get_logger("federation.simulation")


def run_job_in_process(job_path: str, num_supernodes: int | None = None) -> None:
    """Run a job; ``num_supernodes`` > number of banks adds nodes without a bank (tests)."""
    os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    from flwr.simulation import run_simulation

    from heterofl.federation.client_app import app as client_app
    from heterofl.federation.job import load_job
    from heterofl.federation.server_app import make_server_app

    job = load_job(job_path)
    n_nodes = int(num_supernodes or len(job.clients))
    if n_nodes < len(job.clients):
        raise ValueError(f"num_supernodes={n_nodes} < {len(job.clients)} banks")
    total_cpus = max(int(job.num_cpus_per_client) * n_nodes, 1)
    backend_config = {
        "client_resources": {"num_cpus": int(job.num_cpus_per_client),
                             "num_gpus": float(job.num_gpus_per_client)},
        "init_args": {"num_cpus": total_cpus, "include_dashboard": False,
                      "log_to_driver": True},
    }
    run_simulation(server_app=make_server_app(job_path), client_app=client_app,
                   num_supernodes=n_nodes, backend_config=backend_config)
    status = read_json(Path(job.job_dir) / "server" / "status.json")
    if status.get("status") != "completed":
        raise RuntimeError(f"Federated job did not complete: {status}")


def space_free_python() -> str:
    """Path to the current interpreter that contains no whitespace.

    Ray's raylet starts workers through a shell command built from
    ``sys.executable``; a space in the path (e.g. ``~/Documents/Coding Projects``)
    splits it and every worker dies on start. If needed, the whole environment
    prefix (venv) is exposed through a space-free symlink, so the child
    interpreter still finds its ``pyvenv.cfg`` and site-packages.
    """
    exe = Path(sys.executable)
    if not any(c.isspace() for c in str(exe)):
        return str(exe)
    prefix = Path(sys.prefix)
    try:
        rel = exe.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(f"Interpreter {exe} is outside sys.prefix {prefix}; cannot "
                           "build a space-free path for Ray") from exc
    digest = hashlib.sha256(str(prefix).encode()).hexdigest()[:10]
    link = Path(tempfile.gettempdir()) / f"heterofl-env-{digest}"
    if any(c.isspace() for c in str(link)):
        raise RuntimeError(f"Temporary directory {link.parent} contains spaces; set TMPDIR")
    if link.is_symlink() and Path(os.readlink(link)) != prefix:
        link.unlink()
    if not link.exists():
        link.symlink_to(prefix, target_is_directory=True)
    return str(link / rel)


def launch_subprocess(job_path: Path, log_path: Path, timeout: float | None = None,
                      num_supernodes: int | None = None) -> None:
    """Run a job in a fresh interpreter; raise with the log tail on failure.

    POSIX only: the child gets its own session (``start_new_session``) so that on
    timeout or interruption the whole process group -- including Ray's workers --
    is killed with ``os.killpg``.
    """
    # Put this package's source root on PYTHONPATH explicitly, so the child (and
    # the Ray workers, which inherit its environment) can import heterofl even if
    # the editable-install .pth file is skipped (Python >= 3.12 ignores .pth files
    # carrying the macOS "hidden" flag).
    src_root = str(Path(__file__).resolve().parents[2])
    pythonpath = os.pathsep.join(p for p in (src_root, os.environ.get("PYTHONPATH")) if p)
    env = {**os.environ, "FLWR_TELEMETRY_ENABLED": "0", "RAY_DEDUP_LOGS": "0",
           "PYTHONUNBUFFERED": "1", "PYTHONPATH": pythonpath}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        # own session/process group: on timeout the child AND its Ray processes are killed
        cmd = [space_free_python(), "-m", "heterofl.federation.simulation", "--job", str(job_path)]
        if num_supernodes:
            cmd += ["--num-supernodes", str(int(num_supernodes))]
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise RuntimeError(f"Federated simulation exceeded job_timeout={timeout}s and was "
                               f"killed; log: {log_path}") from None
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL)
            raise
    if proc.returncode != 0:
        tail = "".join(log_path.read_text(errors="replace").splitlines(True)[-40:])
        raise RuntimeError(f"Federated simulation failed (exit {proc.returncode}); "
                           f"log: {log_path}\n--- log tail ---\n{tail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", required=True, help="path to job.json")
    parser.add_argument("--num-supernodes", type=int, default=None,
                        help="default: one per bank; more adds nodes without a bank (testing)")
    args = parser.parse_args(argv)
    from heterofl.utils.logging import setup_logging

    setup_logging()
    run_job_in_process(str(Path(args.job).resolve()), args.num_supernodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
