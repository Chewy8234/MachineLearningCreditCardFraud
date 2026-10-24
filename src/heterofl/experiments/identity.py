"""Experiment and run identity: what a stored result was produced from.

A result may only be reused (skipped on resume) when its recorded identity equals
the identity requested now. Two levels are recorded:

``experiment.json`` (results root)
    The semantic configuration (``heterofl.utils.config.semantic_config``: every
    setting that can change an output; locations, timeouts and report-only settings
    excluded), the dataset SHA-256 and the versions of the numerical libraries.
    A results directory belongs to exactly one experiment identity: requesting a
    different one is an error (``ConfigMismatchError``), never a silent overwrite.

``run_identity.json`` (each run directory)
    The run's *effective* configuration (ablation overrides merged), its seed,
    scenario definition, method, variant, and the fingerprint of the partition it
    used. ``run_hash`` = SHA-256 of that record.

Source code is recorded separately (``git_info``), because a changed revision is a
different kind of mismatch: by default it also blocks reuse (``SourceMismatchError``),
and ``--allow-source-change`` reuses such runs while marking them in the outcome log.
"""

from __future__ import annotations

import os
from pathlib import Path

from heterofl.utils.config import config_diff, config_hash, semantic_config, sha256_json

#: Bump when the identity record layout changes; old records then no longer match.
IDENTITY_SCHEMA = 1

#: Scenario keys that only select methods or label reporting rows.
_SCENARIO_SELECTION_KEYS = ("methods", "ablation")


class ResultsIdentityError(RuntimeError):
    """A stored result cannot be reused or extended under the requested identity."""


class ConfigMismatchError(ResultsIdentityError):
    """The stored result was produced under a different configuration."""


class LegacyResultsError(ResultsIdentityError):
    """The stored result predates identity recording; compatibility cannot be shown."""


class SourceMismatchError(ResultsIdentityError):
    """The stored result was produced by a different source revision."""


class SourceNotVerifiedError(ResultsIdentityError):
    """``--require-clean-source`` was given but the working tree is not a clean commit."""


class ResultsLockedError(ResultsIdentityError):
    """Another process is writing to the same results directory."""


def identity_hash(identity: dict) -> str:
    return sha256_json(identity)


def partition_fingerprint(partition_manifest: dict) -> str:
    """Hash of the realised partition: row-id hashes and class counts of every set."""
    return sha256_json({"sets": partition_manifest["sets"],
                        "proxy_row_id_sha256": partition_manifest["proxy_row_id_sha256"],
                        "spec": partition_manifest["spec"]})


def experiment_identity(cfg: dict, *, dataset_sha256: str, packages: dict) -> dict:
    return {"schema": IDENTITY_SCHEMA, "config": semantic_config(cfg),
            "dataset_sha256": dataset_sha256, "packages": dict(packages)}


def run_identity(cfg: dict, *, method: str, seed: int, scenario: str, variant: str,
                 dataset_sha256: str, packages: dict, partition_sha256: str) -> dict:
    sc = {k: v for k, v in cfg["scenarios"][scenario].items() if k not in _SCENARIO_SELECTION_KEYS}
    return {"schema": IDENTITY_SCHEMA, "config": semantic_config(cfg),
            "config_hash": config_hash(cfg), "dataset_sha256": dataset_sha256,
            "packages": dict(packages), "seed": int(seed), "scenario": scenario,
            "scenario_config": sc, "method": method, "variant": variant,
            "partition_sha256": partition_sha256}


def describe_mismatch(stored: dict, requested: dict, limit: int = 12) -> str:
    keys = config_diff(stored, requested)
    shown = ", ".join(keys[:limit]) + (f", ... ({len(keys)} in total)" if len(keys) > limit else "")
    return shown or "<identity layout>"


def dependency_mismatch(dep: dict, own: dict, ignore_sections: tuple[str, ...]) -> list[str]:
    """Differences between the identity of a run we depend on and our own.

    ``ignore_sections`` are config sections the dependency does not read (e.g. the
    local-only models do not read ``ensemble`` or ``kd``); method and variant always
    differ by construction and are ignored.
    """
    def strip(identity: dict) -> dict:
        d = {k: v for k, v in identity.items()
             if k not in ("method", "variant", "config_hash")}
        d["config"] = {k: v for k, v in identity.get("config", {}).items()
                       if k not in ignore_sections}
        return d
    return config_diff(strip(dep), strip(own))


class ResultsLock:
    """Exclusive advisory lock on a results directory (POSIX ``fcntl.flock``).

    Held for the lifetime of a runner invocation, so two processes can never write
    the same results tree. The lock file holds the owner's PID for diagnostics; the
    kernel releases the lock if the process dies, so there is no stale-lock cleanup.
    On platforms without ``fcntl`` (Windows) locking is unavailable and a warning is
    logged by the caller.
    """

    FILENAME = ".heterofl.lock"

    def __init__(self, root: Path):
        self.path = Path(root) / self.FILENAME
        self._fh = None

    def acquire(self) -> bool:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            owner = fh.read().strip() or "unknown"
            fh.close()
            raise ResultsLockedError(f"{self.path.parent} is in use by another process "
                                     f"(lock {self.path}, pid {owner}); wait for it to finish "
                                     "or use a different results_dir") from None
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is not None:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
