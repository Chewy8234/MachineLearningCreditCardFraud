"""Experiment data preparation: processed bundle, partitions, manifests, federation views.

Two levels of cached artifacts are produced deterministically:

``data/processed/<tag>/``  (one per dataset + split configuration)
    X.npy, y.npy, row_ids.npy  -- preprocessed features for every retained row
    splits.npz                 -- test / validation / proxy / client_pool indices
    preprocessor.json          -- scaler statistics (fitted on proxy features only)
    split_manifest.json        -- sizes, class counts, index hashes, overlap checks
    dataset_audit.json

``data/partitions/<tag>/seed_<s>/<scenario>/``  (one per seed x scenario)
    partition.npz, partition_manifest.json, partition_report.csv
    view/  -- the *federation view*: the only data the Flower apps can read
        bank_<X>.npz     private training rows of one bank (features + labels)
        proxy.npz        proxy FEATURES and row ids -- labels are not written
        validation.npz   validation features + labels (evaluation harness)

The test split is never written into a federation view; only the evaluation
harness reads it from the processed bundle after training has finished.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from heterofl.data.loader import (
    FEATURE_COLUMNS,
    dataset_fingerprint,
    features_and_target,
    load_creditcard,
)
from heterofl.data.partitioning import (
    LeakageError,
    assert_disjoint,
    make_partition,
    partition_report,
    split_dataset,
    stratified_subsample,
)
from heterofl.data.preprocessing import FeaturePreprocessor
from heterofl.data.validation import audit_dataset
from heterofl.utils.config import resolve_path
from heterofl.utils.io import load_npz, read_json, save_npz, write_csv, write_json
from heterofl.utils.logging import get_logger
from heterofl.utils.reproducibility import sha256_array

log = get_logger("data.pipeline")

BUNDLE_VERSION = 1


def _config_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def class_counts(y: np.ndarray) -> dict:
    n = int(len(y))
    fraud = int((y == 1).sum())
    return {"n": n, "fraud": fraud, "legitimate": n - fraud,
            "fraud_rate": fraud / n if n else float("nan")}


@dataclass
class DataBundle:
    """Processed dataset + fixed splits (shared by every seed and scenario)."""

    dir: Path

    @cached_property
    def X(self) -> np.ndarray:
        return np.load(self.dir / "X.npy", mmap_mode="r")

    @cached_property
    def y(self) -> np.ndarray:
        return np.load(self.dir / "y.npy")

    @cached_property
    def row_ids(self) -> np.ndarray:
        return np.load(self.dir / "row_ids.npy")

    @cached_property
    def splits(self) -> dict[str, np.ndarray]:
        return load_npz(self.dir / "splits.npz")

    @cached_property
    def manifest(self) -> dict:
        return read_json(self.dir / "split_manifest.json")

    def subset(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        idx = self.splits[name]
        return np.asarray(self.X[idx]), self.y[idx]


@dataclass
class ScenarioData:
    """Partitions + federation view for one (seed, scenario)."""

    dir: Path
    bundle: DataBundle
    banks: list[str]

    @property
    def view_dir(self) -> Path:
        return self.dir / "view"

    @cached_property
    def manifest(self) -> dict:
        return read_json(self.dir / "partition_manifest.json")

    def bank(self, bank: str) -> tuple[np.ndarray, np.ndarray]:
        d = load_npz(self.view_dir / f"bank_{bank}.npz")
        return d["X"], d["y"]

    def proxy_features(self) -> tuple[np.ndarray, np.ndarray]:
        d = load_npz(self.view_dir / "proxy.npz")
        return d["X"], d["row_ids"]

    def validation(self) -> tuple[np.ndarray, np.ndarray]:
        d = load_npz(self.view_dir / "validation.npz")
        return d["X"], d["y"]

    def test(self) -> tuple[np.ndarray, np.ndarray]:
        """Final test split -- for the evaluation harness only, after training."""
        return self.bundle.subset("test")

    def pooled_clients(self) -> tuple[np.ndarray, np.ndarray]:
        """Union of all bank partitions, ordered by row id.

        The union is the same client pool in every scenario, and ordering by row
        id makes the centralized reference identical across scenarios too.
        """
        parts = [load_npz(self.view_dir / f"bank_{b}.npz") for b in self.banks]
        rid = np.concatenate([p["row_ids"] for p in parts])
        order = np.argsort(rid, kind="stable")
        X = np.concatenate([p["X"] for p in parts])[order]
        y = np.concatenate([p["y"] for p in parts])[order]
        return X, y


# ---------------------------------------------------------------------------
# Level 1: processed bundle
# ---------------------------------------------------------------------------

def _bundle_spec(cfg: dict, fingerprint: dict) -> dict:
    data = cfg["data"]
    return {
        "bundle_version": BUNDLE_VERSION,
        "dataset_sha256": fingerprint["sha256"],
        "subsample_fraction": data.get("subsample_fraction"),
        "subsample_seed": data.get("subsample_seed", 0),
        "drop_duplicates": bool(data.get("drop_duplicates", False)),
        "split": {k: cfg["split"][k] for k in ("seed", "test_fraction",
                                              "validation_fraction", "proxy_fraction",
                                              "proxy_sampling")},
        "preprocessing": cfg.get("preprocessing", {"log_amount": True}),
    }


def prepare_bundle(cfg: dict) -> DataBundle:
    """Build (or reuse, if identical) the processed bundle for this config."""
    csv_path = resolve_path(cfg, cfg["data"]["path"])
    df = load_creditcard(csv_path)
    fingerprint = dataset_fingerprint(csv_path)
    spec = _bundle_spec(cfg, fingerprint)
    tag = f"{'synthetic' if cfg['data'].get('synthetic') else 'ulb'}-{_config_hash(spec)}"
    out = resolve_path(cfg, cfg["data"]["processed_dir"]) / tag
    done = out / "split_manifest.json"
    if done.is_file() and read_json(done).get("spec") == spec:
        log.info("Reusing processed bundle %s", out)
        return DataBundle(out)

    log.info("Building processed bundle %s", out)
    audit = audit_dataset(df, expect_ulb=not cfg["data"].get("synthetic", False))
    audit["fingerprint"] = fingerprint
    audit["synthetic"] = bool(cfg["data"].get("synthetic", False))

    row_ids = np.arange(len(df))
    if spec["drop_duplicates"]:
        keep = ~df.duplicated(keep="first").to_numpy()
        log.warning("data.drop_duplicates=true: removing %d exact duplicate rows "
                    "(explicitly requested)", int((~keep).sum()))
        row_ids = row_ids[keep]
    X_raw, y_all = features_and_target(df)
    X_raw, y = X_raw[row_ids], y_all[row_ids]

    candidate = np.arange(len(y))
    if spec["subsample_fraction"] not in (None, 1, 1.0):
        candidate = stratified_subsample(y, float(spec["subsample_fraction"]),
                                         int(spec["subsample_seed"]))
        log.warning("Stratified subsample %.3f -> %d rows (development use only)",
                    spec["subsample_fraction"], len(candidate))

    s = cfg["split"]
    splits = split_dataset(y, test_fraction=s["test_fraction"],
                           validation_fraction=s["validation_fraction"],
                           proxy_fraction=s["proxy_fraction"],
                           proxy_sampling=s.get("proxy_sampling", "stratified"),
                           seed=int(s["seed"]), candidate_idx=candidate)

    pre = FeaturePreprocessor(log_amount=spec["preprocessing"].get("log_amount", True))
    pre.fit(X_raw[splits.proxy], source="proxy split features (labels unused)")
    X = pre.transform(X_raw)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    np.save(out / "row_ids.npy", row_ids)
    save_npz(out / "splits.npz", **splits.as_dict())
    write_json(out / "preprocessor.json", pre.to_dict())
    write_json(out / "dataset_audit.json", audit)

    sets = splits.as_dict()
    dup = _cross_split_duplicates(X_raw, sets)
    manifest = {
        "spec": spec,
        "tag": tag,
        "synthetic": audit["synthetic"],
        "n_rows_loaded": int(len(df)),
        "n_rows_retained": int(len(y)),
        "n_rows_in_experiment": int(len(candidate)),
        "split_seed": int(s["seed"]),
        "proxy_construction": (
            "stratified on labels before labels were dropped (label-informed reference "
            "set construction); proxy labels are never written to the federation view"
            if s.get("proxy_sampling", "stratified") == "stratified"
            else "uniform random sample (natural class composition); labels never shared"),
        "splits": {name: {**class_counts(y[idx]),
                          "index_sha256": sha256_array(np.asarray(idx, dtype=np.int64)),
                          "row_id_sha256": sha256_array(row_ids[idx].astype(np.int64))}
                   for name, idx in sets.items()},
        "overlap_check": _overlap_matrix(sets),
        "cross_split_exact_feature_duplicates": dup,
        "preprocessing": {k: v for k, v in pre.to_dict().items()
                          if k not in ("mean", "scale")},
        "dataset": fingerprint,
    }
    write_json(done, manifest)
    return DataBundle(out)


def _overlap_matrix(sets: dict[str, np.ndarray]) -> dict:
    names = list(sets)
    res = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res[f"{a}&{b}"] = int(np.intersect1d(sets[a], sets[b]).size)
    res["all_disjoint"] = all(v == 0 for v in res.values())
    return res


def _cross_split_duplicates(X_raw: np.ndarray, sets: dict[str, np.ndarray]) -> dict:
    """Count evaluation rows whose exact raw feature vector also occurs in a training split.

    These are genuine duplicates in the ULB data (not leakage introduced by the
    pipeline); they are reported so readers can judge their effect.
    """
    hashes = pd.util.hash_pandas_object(pd.DataFrame(X_raw), index=False).to_numpy()
    train_like = np.concatenate([sets["client_pool"], sets["proxy"]])
    train_hashes = set(hashes[train_like].tolist())
    return {name: int(sum(h in train_hashes for h in hashes[sets[name]].tolist()))
            for name in ("validation", "test")}


# ---------------------------------------------------------------------------
# Level 2: per-seed, per-scenario partitions and federation view
# ---------------------------------------------------------------------------

def prepare_scenario(cfg: dict, bundle: DataBundle, seed: int, scenario_name: str) -> ScenarioData:
    scenario = cfg["scenarios"][scenario_name]
    part_cfg = cfg["partitioning"]
    banks = [c["bank"] for c in cfg["clients"]]
    spec = {
        "bundle_tag": bundle.manifest["tag"],
        "seed": int(seed),
        "scenario": scenario_name,
        "scenario_config": scenario,
        "num_clients": int(part_cfg["num_clients"]),
        "min_fraud_per_client": int(part_cfg["min_fraud_per_client"]),
        "min_legit_per_client": int(part_cfg["min_legit_per_client"]),
        "max_retries": int(part_cfg["max_retries"]),
        "banks": banks,
    }
    out = (resolve_path(cfg, cfg["data"]["partitions_dir"]) / bundle.manifest["tag"]
           / f"seed_{seed}" / scenario_name)
    done = out / "partition_manifest.json"
    if done.is_file() and read_json(done).get("spec") == spec:
        log.info("Reusing partition %s", out)
        return ScenarioData(out, bundle, banks)

    y = bundle.y
    sp = bundle.splits
    parts, info = make_partition(sp["client_pool"], y, scenario, spec["num_clients"], seed,
                                 min_fraud=spec["min_fraud_per_client"],
                                 min_legit=spec["min_legit_per_client"],
                                 max_retries=spec["max_retries"])
    sets = {f"bank_{b}": p for b, p in zip(banks, parts, strict=True)}
    sets.update({k: sp[k] for k in ("proxy", "validation", "test")})
    assert_disjoint(sets)

    out.mkdir(parents=True, exist_ok=True)
    save_npz(out / "partition.npz", **{f"bank_{b}": p for b, p in zip(banks, parts, strict=True)})
    rows = partition_report(parts, y, banks, seed, scenario_name, scenario)
    write_csv(out / "partition_report.csv", rows)

    view = out / "view"
    X = bundle.X
    rid = bundle.row_ids
    view_files = {}
    for b, p in zip(banks, parts, strict=True):
        save_npz(view / f"bank_{b}.npz", X=np.asarray(X[p]), y=y[p], row_ids=rid[p])
        view_files[f"bank_{b}"] = sha256_array(rid[p].astype(np.int64))
    # Proxy: features + identifiers only. Labels are deliberately not saved.
    save_npz(view / "proxy.npz", X=np.asarray(X[sp["proxy"]]), row_ids=rid[sp["proxy"]])
    save_npz(view / "validation.npz", X=np.asarray(X[sp["validation"]]),
             y=y[sp["validation"]], row_ids=rid[sp["validation"]])

    manifest = {
        "spec": spec,
        "partition_info": info,
        "clients": rows,
        "sets": {name: {**class_counts(y[idx]),
                        "row_id_sha256": sha256_array(rid[idx].astype(np.int64))}
                 for name, idx in sets.items()},
        "proxy_row_id_sha256": sha256_array(rid[sp["proxy"]].astype(np.int64)),
        "overlap_check": _overlap_matrix(sets),
        "coverage_check": bool(np.array_equal(np.sort(np.concatenate(parts)),
                                              np.sort(sp["client_pool"]))),
        "view_contents": {
            "bank_<X>.npz": "X, y, row_ids of that bank's private training partition",
            "proxy.npz": "X, row_ids only (no labels)",
            "validation.npz": "X, y, row_ids (evaluation harness; threshold/model selection)",
            "test": "NOT present in the view",
        },
        "bundle_split_manifest": str(bundle.dir / "split_manifest.json"),
    }
    if not manifest["overlap_check"]["all_disjoint"] or not manifest["coverage_check"]:
        raise LeakageError(f"Partition validation failed for {scenario_name}/seed {seed}")
    write_json(done, manifest)
    log.info("Partition %s seed=%d: %s", scenario_name, seed,
             ", ".join(f"{r['bank']}={r['n_transactions']}/{r['n_fraud']}f" for r in rows))
    return ScenarioData(out, bundle, banks)


FEATURE_NAMES = list(FEATURE_COLUMNS)
