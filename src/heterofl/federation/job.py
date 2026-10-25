"""Federated job specification shared by the ServerApp and the ClientApps.

The experiment harness writes ``job.json`` into the job directory; both apps
read it (the path travels in the run config / message ConfigRecord). The job
only references the *federation view* directory, which contains each bank's
private partition, proxy features without labels, and validation data; the
test split is not reachable from a job.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from heterofl.utils.io import read_json, write_json


class FederationError(RuntimeError):
    """A federated round failed (missing reply, client error, invalid payload)."""


@dataclass
class ClientSpec:
    partition_id: int
    bank: str
    model_type: str


@dataclass
class FederatedJob:
    mode: str                      # "fedavg" | "kd"
    job_dir: str
    view_dir: str
    seed: int
    num_rounds: int
    clients: list[ClientSpec]
    models: dict
    fedavg: dict
    kd: dict
    expected_proxy_sha256: str
    num_cpus_per_client: int = 1
    num_gpus_per_client: float = 0.0
    round_timeout: float = 3600.0
    node_wait_timeout: float = 120.0
    debug: dict = field(default_factory=dict)

    def __post_init__(self):
        self.clients = [c if isinstance(c, ClientSpec) else ClientSpec(**c) for c in self.clients]
        if self.mode not in ("fedavg", "kd"):
            raise ValueError(f"Unknown federated mode {self.mode!r}")
        pids = sorted(c.partition_id for c in self.clients)
        if pids != list(range(len(self.clients))):
            raise ValueError(f"Client partition ids must be 0..K-1, got {pids}")

    def find_client(self, partition_id: int) -> ClientSpec | None:
        """The bank assigned to ``partition_id``, or ``None`` if the node has no bank."""
        for c in self.clients:
            if c.partition_id == partition_id:
                return c
        return None

    def client(self, partition_id: int) -> ClientSpec:
        spec = self.find_client(partition_id)
        if spec is None:
            raise FederationError(f"No client is assigned to partition-id {partition_id}")
        return spec

    @property
    def torch_threads(self) -> int:
        return int((self.models.get("mlp") or {}).get("torch_threads", 1))

    def model_type_for(self, spec: ClientSpec) -> str:
        return "mlp" if self.mode == "fedavg" else spec.model_type

    @property
    def path(self) -> Path:
        return Path(self.job_dir) / "job.json"

    def save(self) -> Path:
        write_json(self.path, asdict(self))
        return self.path


JOB_FILENAME = "job.json"


def check_job_path(path: str | Path) -> Path:
    """Validate a ``job-path`` received in a Flower message before reading it.

    The path must be absolute, name a ``job.json`` file that exists, and -- if the
    environment variable ``HETEROFL_JOB_ROOT`` is set -- lie inside that directory.
    This is a guard against accidental misuse, not a security boundary: the
    simulation assumes a trusted server and trusted local files (docs/reproducibility.md).
    """
    p = Path(str(path))
    if not p.is_absolute() or p.name != JOB_FILENAME:
        raise FederationError(f"job-path must be an absolute path to {JOB_FILENAME}, got {path!r}")
    p = p.resolve()
    root = os.environ.get("HETEROFL_JOB_ROOT")
    if root and not p.is_relative_to(Path(root).resolve()):
        raise FederationError(f"job-path {p} is outside HETEROFL_JOB_ROOT={root}")
    if not p.is_file():
        raise FederationError(f"job-path {p} does not exist")
    return p


@lru_cache(maxsize=8)
def load_job(path: str) -> FederatedJob:
    return FederatedJob(**read_json(check_job_path(path)))
