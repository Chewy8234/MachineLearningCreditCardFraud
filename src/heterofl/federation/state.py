"""Client model state that survives across rounds and simulation worker lifecycles.

Flower simulation keeps a per-node ``Context`` whose ``state`` RecordDict is
carried between messages, but the Python process executing a ClientApp
(a Ray actor) is not tied to a node. Model objects are therefore persisted
to disk per (bank, round), and ``context.state`` records the round number,
file path and SHA-256 of the latest checkpoint. Every distillation round
verifies that record against the file before training, so lost or
mismatched state raises ``ClientStateError`` instead of silently restarting
from scratch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from flwr.app import ConfigRecord, Context

from heterofl.models.base import FraudModel
from heterofl.models.model_factory import load_model
from heterofl.utils.io import write_json
from heterofl.utils.reproducibility import sha256_file

STATE_KEY = "heterofl-client-state"


class ClientStateError(RuntimeError):
    """Client state missing, from the wrong round, or inconsistent with disk."""


class ClientStateStore:
    def __init__(self, job_dir: str | Path, bank: str):
        self.dir = Path(job_dir) / "clients" / f"bank_{bank}"
        self.bank = bank

    def round_dir(self, rnd: int) -> Path:
        return self.dir / f"round_{rnd:02d}"

    def save_round(self, model: FraudModel, rnd: int, *, val_probs: np.ndarray,
                   proxy_probs: np.ndarray, teacher: np.ndarray | None, info: dict) -> tuple[Path, str]:
        d = self.round_dir(rnd)
        d.mkdir(parents=True, exist_ok=True)
        path = model.save(d / "model")
        digest = sha256_file(path)
        np.save(d / "val_probs.npy", np.asarray(val_probs, dtype=np.float64))
        np.save(d / "proxy_probs.npy", np.asarray(proxy_probs, dtype=np.float64))
        if teacher is not None:
            np.save(d / "teacher.npy", np.asarray(teacher, dtype=np.float64))
        write_json(d / "meta.json", {"bank": self.bank, "round": rnd,
                                     "model_type": model.model_type, "model_file": path.name,
                                     "model_sha256": digest, "fit_info": info})
        return path, digest

    @staticmethod
    def record(context: Context, *, bank: str, rnd: int, path: Path, digest: str,
               model_type: str) -> None:
        context.state[STATE_KEY] = ConfigRecord({
            "bank": bank, "round": int(rnd), "model-path": str(path),
            "model-sha256": digest, "model-type": model_type})

    def load_previous(self, context: Context, *, expected_round: int, model_type: str,
                      models_cfg: dict, seed: int, n_jobs: int) -> FraudModel:
        if STATE_KEY not in context.state:
            raise ClientStateError(f"bank {self.bank}: no client state in context; expected "
                                   f"round {expected_round} checkpoint")
        st = context.state[STATE_KEY]
        if st["bank"] != self.bank:
            raise ClientStateError(f"context state belongs to bank {st['bank']}, not {self.bank}")
        if int(st["round"]) != expected_round:
            raise ClientStateError(f"bank {self.bank}: state is at round {st['round']}, "
                                   f"expected {expected_round}")
        path = Path(str(st["model-path"])).resolve()
        # Only this bank's own checkpoints (written by save_round) are ever loaded.
        if not path.is_relative_to(self.dir.resolve()):
            raise ClientStateError(f"bank {self.bank}: checkpoint {path} is outside this "
                                   f"bank's state directory {self.dir}; refusing to load it")
        if not path.is_file():
            raise ClientStateError(f"bank {self.bank}: checkpoint {path} is missing")
        if sha256_file(path) != st["model-sha256"]:
            raise ClientStateError(f"bank {self.bank}: checkpoint {path} hash mismatch")
        return load_model(model_type, path, models_cfg, seed, n_jobs)
