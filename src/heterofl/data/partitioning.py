"""Train/validation/proxy/test splitting and IID / non-IID client partitioning.

Split protocol (all index arrays refer to rows of the loaded dataset):

    full dataset (optionally stratified-subsampled for smoke tests)
      |-- test        : ``test_fraction`` (stratified)
      `-- development : the rest
            |-- validation : ``validation_fraction`` of development (stratified)
            `-- dev-train  : the rest
                  |-- proxy       : ``proxy_fraction`` of dev-train
                  |                 (stratified = label-informed construction, or random)
                  `-- client pool : the rest, partitioned among the banks

Client partitioning:

* ``iid``       : per-class shuffling and equal splitting, so every bank gets an
                  (almost) equal share of each class.
* ``dirichlet`` : class-conditioned Dirichlet partitioning. For each class c,
                  proportions q_c ~ Dir(alpha * 1_K) are drawn and class-c rows are
                  split accordingly. Draws violating the minimum per-client
                  class counts are rejected and redrawn (bounded retries). The
                  accepted partition is therefore a draw from the Dirichlet
                  scheme *conditioned on the constraints*; the actual achieved
                  distributions are reported, because alpha alone does not fix
                  the degree of heterogeneity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from heterofl.utils.reproducibility import derive_seed


class PartitionError(RuntimeError):
    """A requested partition configuration could not be satisfied."""


class LeakageError(AssertionError):
    """Index sets that must be disjoint overlap."""


@dataclass
class SplitIndices:
    test: np.ndarray
    validation: np.ndarray
    proxy: np.ndarray
    client_pool: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"test": self.test, "validation": self.validation,
                "proxy": self.proxy, "client_pool": self.client_pool}


def stratified_subsample(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Row indices of a stratified subsample (used only for smoke profiles)."""
    if not 0 < fraction <= 1:
        raise ValueError("subsample fraction must be in (0, 1]")
    idx = np.arange(len(y))
    if fraction == 1:
        return idx
    keep, _ = train_test_split(idx, train_size=fraction, stratify=y, random_state=seed)
    return np.sort(keep)


def split_dataset(y: np.ndarray, *, test_fraction: float, validation_fraction: float,
                  proxy_fraction: float, proxy_sampling: str, seed: int,
                  candidate_idx: np.ndarray | None = None) -> SplitIndices:
    """Create the four mutually disjoint index sets (see module docstring)."""
    idx = np.arange(len(y)) if candidate_idx is None else np.asarray(candidate_idx)
    yy = y[idx]
    dev, test = train_test_split(idx, test_size=test_fraction, stratify=yy,
                                 random_state=derive_seed(seed, "split", "test"))
    dev_train, val = train_test_split(dev, test_size=validation_fraction, stratify=y[dev],
                                      random_state=derive_seed(seed, "split", "validation"))
    if proxy_sampling == "stratified":
        strat = y[dev_train]
    elif proxy_sampling == "random":
        strat = None
    else:
        raise ValueError(f"Unknown proxy_sampling {proxy_sampling!r}")
    pool, proxy = train_test_split(dev_train, test_size=proxy_fraction, stratify=strat,
                                   random_state=derive_seed(seed, "split", "proxy"))
    out = SplitIndices(test=np.sort(test), validation=np.sort(val),
                       proxy=np.sort(proxy), client_pool=np.sort(pool))
    assert_disjoint(out.as_dict())
    return out


def assert_disjoint(sets: dict[str, np.ndarray]) -> None:
    """Raise LeakageError if any two named index sets share an element."""
    names = list(sets)
    for i, a in enumerate(names):
        if len(np.unique(sets[a])) != len(sets[a]):
            raise LeakageError(f"Index set {a!r} contains duplicated indices")
        for b in names[i + 1:]:
            inter = np.intersect1d(sets[a], sets[b], assume_unique=True)
            if inter.size:
                raise LeakageError(f"Index sets {a!r} and {b!r} overlap in "
                                   f"{inter.size} rows (e.g. {inter[:5].tolist()})")


def partition_iid(pool_idx: np.ndarray, y: np.ndarray, num_clients: int,
                  seed: int) -> list[np.ndarray]:
    """Stratified equal partition: each class is shuffled and split into K equal parts."""
    rng = np.random.default_rng(derive_seed(seed, "partition", "iid"))
    parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for cls in (0, 1):
        cls_idx = pool_idx[y[pool_idx] == cls]
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        # rotate which client receives the +1 remainder so sizes stay balanced
        chunks = np.array_split(cls_idx, num_clients)
        offset = cls % num_clients
        for k in range(num_clients):
            parts[k].append(chunks[(k + offset) % num_clients])
    return [np.sort(np.concatenate(p)) for p in parts]


def _dirichlet_draw(pool_idx: np.ndarray, y: np.ndarray, num_clients: int, alpha: float,
                    rng: np.random.Generator) -> list[np.ndarray]:
    parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for cls in (0, 1):
        cls_idx = pool_idx[y[pool_idx] == cls]
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        q = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(q)[:-1] * len(cls_idx)).astype(int)
        for k, chunk in enumerate(np.split(cls_idx, cuts)):
            parts[k].append(chunk)
    return [np.sort(np.concatenate(p)) for p in parts]


def partition_dirichlet(pool_idx: np.ndarray, y: np.ndarray, num_clients: int,
                        alpha: float, seed: int, *, min_fraud: int, min_legit: int,
                        max_retries: int) -> tuple[list[np.ndarray], dict]:
    """Class-conditioned Dirichlet partition with bounded, validated retries.

    Returns ``(parts, info)`` where ``info`` records the accepted attempt and the
    number of rejected draws. Raises PartitionError if no draw within
    ``max_retries`` satisfies the constraints (the design is never silently
    relaxed).
    """
    n_fraud = int((y[pool_idx] == 1).sum())
    n_legit = int((y[pool_idx] == 0).sum())
    _check_feasible(n_fraud, n_legit, num_clients, min_fraud, min_legit)

    best = None
    for attempt in range(max_retries):
        rng = np.random.default_rng(derive_seed(seed, "partition", "dirichlet", alpha, attempt))
        parts = _dirichlet_draw(pool_idx, y, num_clients, alpha, rng)
        fraud = [int((y[p] == 1).sum()) for p in parts]
        legit = [int((y[p] == 0).sum()) for p in parts]
        if min(fraud) >= min_fraud and min(legit) >= min_legit:
            return parts, {"accepted_attempt": attempt, "rejected_draws": attempt,
                           "max_retries": max_retries}
        score = min(min(fraud) / max(min_fraud, 1), min(legit) / max(min_legit, 1))
        if best is None or score > best[0]:
            best = (score, fraud, legit)
    raise PartitionError(
        f"Dirichlet(alpha={alpha}) partitioning failed: no draw in {max_retries} attempts "
        f"gave every client >= {min_fraud} fraud and >= {min_legit} legitimate rows "
        f"(pool has {n_fraud} fraud / {n_legit} legitimate). Closest draw: fraud={best[1]}, "
        f"legit={best[2]}. Increase alpha or max_retries, or lower the minimums in the "
        f"config -- the design is not altered automatically.")


def _check_feasible(n_fraud: int, n_legit: int, k: int, min_fraud: int, min_legit: int) -> None:
    if n_fraud < k * min_fraud:
        raise PartitionError(f"Infeasible: {n_fraud} fraud rows cannot give {k} clients "
                             f">= {min_fraud} each")
    if n_legit < k * min_legit:
        raise PartitionError(f"Infeasible: {n_legit} legitimate rows cannot give {k} "
                             f"clients >= {min_legit} each")


def make_partition(pool_idx: np.ndarray, y: np.ndarray, scenario: dict, num_clients: int,
                   seed: int, *, min_fraud: int, min_legit: int,
                   max_retries: int) -> tuple[list[np.ndarray], dict]:
    """Dispatch on ``scenario['type']`` and validate the result."""
    if scenario["type"] == "iid":
        _check_feasible(int((y[pool_idx] == 1).sum()), int((y[pool_idx] == 0).sum()),
                        num_clients, min_fraud, min_legit)
        parts = partition_iid(pool_idx, y, num_clients, seed)
        info = {"accepted_attempt": 0, "rejected_draws": 0, "max_retries": 1}
        for k, p in enumerate(parts):
            if (y[p] == 1).sum() < min_fraud or (y[p] == 0).sum() < min_legit:
                raise PartitionError(f"IID partition gives client {k} too few examples")
    elif scenario["type"] == "dirichlet":
        parts, info = partition_dirichlet(pool_idx, y, num_clients, float(scenario["alpha"]),
                                          seed, min_fraud=min_fraud, min_legit=min_legit,
                                          max_retries=max_retries)
    else:
        raise ValueError(f"Unknown scenario type {scenario['type']!r}")

    named = {f"client_{k}": p for k, p in enumerate(parts)}
    assert_disjoint(named)
    union = np.sort(np.concatenate(parts))
    if not np.array_equal(union, np.sort(pool_idx)):
        raise LeakageError("Client partitions do not exactly cover the client pool")
    return parts, info


def partition_report(parts: list[np.ndarray], y: np.ndarray, banks: list[str],
                     seed: int, scenario_name: str, scenario: dict) -> list[dict]:
    total = sum(len(p) for p in parts)
    total_fraud = sum(int((y[p] == 1).sum()) for p in parts)
    rows = []
    for bank, p in zip(banks, parts, strict=True):
        fraud = int((y[p] == 1).sum())
        rows.append({
            "scenario": scenario_name,
            "scenario_type": scenario["type"],
            "alpha": scenario.get("alpha"),
            "partition_seed": seed,
            "bank": bank,
            "n_transactions": int(len(p)),
            "n_fraud": fraud,
            "n_legitimate": int(len(p) - fraud),
            "fraud_pct": 100.0 * fraud / max(len(p), 1),
            "relative_size": len(p) / max(total, 1),
            "share_of_fraud": fraud / max(total_fraud, 1),
        })
    return rows
