"""Seeding, hashing and environment capture for reproducible runs."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

TRACKED_PACKAGES = ("flwr", "ray", "xgboost", "torch", "scikit-learn", "pandas",
                    "numpy", "scipy", "matplotlib", "seaborn", "pyyaml", "joblib")


def derive_seed(base_seed: int, *keys: object) -> int:
    """Deterministic 32-bit seed from a base seed and string-able keys.

    Uses SHA-256 rather than ``hash()`` so it is stable across processes
    (Python string hashing is randomised per interpreter).
    """
    text = "|".join([str(base_seed), *map(str, keys)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def git_info(root: str | Path) -> dict:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                 text=True, timeout=10, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def environment_info(root: str | Path | None = None) -> dict:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    info = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
    }
    try:
        import torch

        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_mps_available"] = torch.backends.mps.is_available()
    except ImportError:  # pragma: no cover
        pass
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        info["memory_bytes"] = int(mem)
    except (ValueError, OSError, AttributeError):
        info["memory_bytes"] = None
    if root is not None:
        info["git"] = git_info(root)
    return info
