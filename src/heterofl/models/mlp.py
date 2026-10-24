"""Bank C (and every FedAvg client): a PyTorch multilayer perceptron.

The network outputs a single logit; probabilities are ``sigmoid(logit)``.
All losses use ``binary_cross_entropy_with_logits`` for numerical stability.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from heterofl.models.base import (
    FraudModel,
    check_probabilities,
    int_range,
    real_range,
    validate_params,
)
from heterofl.utils.reproducibility import configure_torch_threads, derive_seed

# Process-wide torch setting, applied ONCE when this module is first imported (not
# per model construction): a single intra-op thread. With XGBoost's OpenMP runtime
# loaded in the same process, multi-threaded torch deadlocks on macOS arm64
# (reproduced with torch 2.14.1 / xgboost 3.4.1); the network is small, so this
# also keeps CPU runs deterministic. Entry points (runner, Flower client/server
# apps) re-apply ``models.mlp.torch_threads`` explicitly via ``configure_torch_threads``.
configure_torch_threads(1)


def _hidden_sizes(v):
    if not isinstance(v, (list, tuple)) or any(int_range(1)(h) for h in v):
        return "must be a list of positive integers"
    return None


def _device(v):
    return None if isinstance(v, str) and v else "must be a torch device string (cpu, cuda, auto)"


PARAM_SPECS = {
    "hidden_sizes": _hidden_sizes,
    "dropout": real_range(0, 1, high_open=True),
    "learning_rate": real_range(0, low_open=True),
    "weight_decay": real_range(0),
    "batch_size": int_range(1),
    "epochs": int_range(1),
    "device": _device,
    "torch_threads": int_range(1),
}
REQUIRED_PARAMS = ("hidden_sizes", "learning_rate", "batch_size", "epochs")


class MLPNet(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class MLPModel(FraudModel):
    model_type = "mlp"
    file_suffix = ".pt"

    @staticmethod
    def validate_params(params: dict) -> None:
        validate_params("mlp", params, PARAM_SPECS, REQUIRED_PARAMS)

    def __init__(self, params: dict, seed: int, n_features: int = 30, n_jobs: int = 1):
        self.validate_params(params)
        self.params = dict(params)
        self.seed = int(seed)
        self.n_features = int(n_features)
        self.device = resolve_device(self.params.get("device", "cpu"))
        self.n_train_calls = 0
        # n_jobs is deliberately not used for torch (see configure_torch_threads above).
        # Initialisation draws from torch's global RNG (nn.Linear has no generator
        # argument), so it runs inside a forked RNG scope: the weights are exactly what
        # ``torch.manual_seed(derive_seed(seed, "mlp-init"))`` produces, and the caller's
        # global RNG state is left untouched.
        with self._rng_scope():
            torch.manual_seed(derive_seed(self.seed, "mlp-init"))
            self.net = MLPNet(self.n_features, list(self.params["hidden_sizes"]),
                              float(self.params.get("dropout", 0.0))).to(self.device)

    def _rng_scope(self):
        """Fork torch's global RNG (CPU, plus the CUDA device in use) and restore it on exit."""
        if self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            return torch.random.fork_rng(devices=[index], device_type="cuda")
        return torch.random.fork_rng(devices=[])

    # ------------------------------------------------------------------ params
    def get_state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in self.net.state_dict().items())

    def set_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state, strict=True)

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    # ------------------------------------------------------------------ training
    def _optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.net.parameters(), lr=float(self.params["learning_rate"]),
                                weight_decay=float(self.params.get("weight_decay", 0.0)))

    def _generator(self) -> torch.Generator:
        """Shuffling generator for one training call; also seeds dropout.

        Dropout has no generator argument and draws from torch's global RNG, so it is
        seeded per training call here. Callers must be inside ``_rng_scope()`` so the
        seeding does not leak into the rest of the process.
        """
        g = torch.Generator()
        g.manual_seed(derive_seed(self.seed, "mlp-train", self.n_train_calls))
        torch.manual_seed(derive_seed(self.seed, "mlp-dropout", self.n_train_calls))
        self.n_train_calls += 1
        return g

    def _tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def fit(self, X, y, sample_weight=None, epochs: int | None = None) -> dict:
        """Supervised training (continues from the current weights, fresh optimizer).

        A newly constructed model starts from its seeded initialisation, so calling
        ``fit`` once on a new model is ordinary from-scratch training.
        """
        return self.fit_distill(X, y, None, None, 0.0, epochs=epochs, sample_weight=sample_weight)

    def fit_distill(self, X, y, X_proxy, teacher, lambda_kd: float, *, epochs: int | None = None,
                    proxy_batch_size: int | None = None, prob_clip: float = 1e-6,
                    sample_weight=None) -> dict:
        """Minimise ``L_sup + lambda_kd * L_kd`` for ``epochs`` passes over the local data.

        * ``L_sup``: mean BCE over a mini-batch of private labelled examples.
        * ``L_kd`` : mean soft-target BCE over a mini-batch of proxy examples, with
          targets = teacher probabilities clipped to ``[prob_clip, 1 - prob_clip]``.

        Both terms are per-example means, so their relative weight is exactly
        ``lambda_kd`` regardless of how many proxy vs. local rows exist. One proxy
        mini-batch is drawn (cycling through a shuffled proxy order) per local step.
        """
        with self._rng_scope():
            return self._fit_distill(X, y, X_proxy, teacher, lambda_kd, epochs=epochs,
                                     proxy_batch_size=proxy_batch_size, prob_clip=prob_clip,
                                     sample_weight=sample_weight)

    def _fit_distill(self, X, y, X_proxy, teacher, lambda_kd: float, *, epochs, proxy_batch_size,
                     prob_clip, sample_weight) -> dict:
        epochs = int(epochs if epochs is not None else self.params["epochs"])
        bs = int(self.params["batch_size"])
        pbs = int(proxy_batch_size or bs)
        use_kd = lambda_kd > 0 and X_proxy is not None
        t0 = time.perf_counter()
        g = self._generator()
        Xt, yt = self._tensor(X), self._tensor(y)
        wt = None if sample_weight is None else self._tensor(sample_weight)
        if use_kd:
            Xp = self._tensor(X_proxy)
            tp = self._tensor(np.clip(np.asarray(teacher, dtype=np.float64), prob_clip, 1 - prob_clip))
            if len(Xp) != len(tp):
                raise ValueError("X_proxy and teacher lengths differ")
        opt = self._optimizer()
        self.net.train()
        n = len(Xt)
        steps = 0
        sup_hist, kd_hist = [], []
        proxy_order, proxy_pos = None, 0
        for _ in range(epochs):
            order = torch.randperm(n, generator=g)
            sup_sum, kd_sum, nb = 0.0, 0.0, 0
            for start in range(0, n, bs):
                idx = order[start:start + bs].to(self.device)
                logits = self.net(Xt[idx])
                l_sup = F.binary_cross_entropy_with_logits(
                    logits, yt[idx], weight=None if wt is None else wt[idx])
                loss = l_sup
                if use_kd:
                    if proxy_order is None or proxy_pos + pbs > len(Xp):
                        proxy_order, proxy_pos = torch.randperm(len(Xp), generator=g), 0
                    pidx = proxy_order[proxy_pos:proxy_pos + pbs].to(self.device)
                    proxy_pos += pbs
                    l_kd = F.binary_cross_entropy_with_logits(self.net(Xp[pidx]), tp[pidx])
                    loss = l_sup + lambda_kd * l_kd
                    kd_sum += float(l_kd.detach())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sup_sum += float(l_sup.detach())
                nb += 1
                steps += 1
            sup_hist.append(sup_sum / max(nb, 1))
            kd_hist.append(kd_sum / max(nb, 1) if use_kd else None)
        self.net.eval()
        return {"fit_seconds": time.perf_counter() - t0, "n_rows": int(n), "epochs": epochs,
                "steps": steps, "final_supervised_loss": sup_hist[-1] if sup_hist else None,
                "final_distillation_loss": kd_hist[-1] if kd_hist else None,
                "supervised_loss_per_epoch": sup_hist,
                "distillation_loss_per_epoch": kd_hist if use_kd else [],
                "lambda_kd": float(lambda_kd) if use_kd else 0.0,
                "update": "continue"}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        out = []
        Xt = self._tensor(X)
        for start in range(0, len(Xt), 8192):
            out.append(torch.sigmoid(self.net(Xt[start:start + 8192])).double().cpu().numpy())
        p = np.concatenate(out) if out else np.zeros(0)
        return check_probabilities(p, len(X), "mlp proba")

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> Path:
        path = self._with_suffix(path)
        torch.save({"state_dict": self.get_state_dict(), "params": self.params,
                    "seed": self.seed, "n_features": self.n_features,
                    "n_train_calls": self.n_train_calls}, path)
        return path

    @classmethod
    def load(cls, path: Path, params: dict | None = None, seed: int | None = None,
             n_jobs: int = 1) -> "MLPModel":
        """Load a checkpoint written by ``save``.

        ``weights_only=True`` restricts unpickling to tensors and plain containers, so a
        tampered file cannot execute code. Checkpoints written by earlier versions of
        this project (same dict layout) load unchanged.
        """
        blob = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(blob["params"], blob["seed"], blob["n_features"], n_jobs)
        obj.set_state_dict(blob["state_dict"])
        obj.n_train_calls = int(blob.get("n_train_calls", 0))
        return obj
