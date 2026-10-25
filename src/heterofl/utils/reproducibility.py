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


def configure_torch_threads(n: int = 1) -> None:
    """Set torch's process-wide intra-op thread count (idempotent).

    Called once at import of ``heterofl.models.mlp`` and at process entry points
    (runner, Flower apps) -- never from a model constructor.
    """
    import torch

    n = int(n)
    if torch.get_num_threads() != n:
        torch.set_num_threads(n)


def _git(root: str | Path, *args: str, text: bool = True):
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                             text=text, timeout=30, check=True)
        return out.stdout.strip() if text else out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(root: str | Path) -> dict:
    """Source revision of ``root``: commit, branch, dirty flag and a fingerprint of local edits.

    ``diff_sha256`` hashes ``git diff HEAD`` (tracked changes, binary-safe) plus the
    names and contents of untracked, non-ignored files, so two dirty trees with the
    same commit can still be told apart. It is ``None`` for a clean tree.
    ``source_verified`` is true only when the code that runs is exactly a commit.
    """
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    commit = _git(root, "rev-parse", "HEAD")
    dirty = bool(status) if status is not None else None
    diff_sha = None
    if dirty:
        h = hashlib.sha256()
        h.update(_git(root, "diff", "HEAD", "--binary", text=False) or b"")
        untracked = _git(root, "ls-files", "--others", "--exclude-standard") or ""
        for name in sorted(filter(None, untracked.splitlines())):
            h.update(b"\0untracked\0" + name.encode())
            try:
                h.update(Path(root, name).read_bytes())
            except OSError:
                h.update(b"<unreadable>")
        diff_sha = h.hexdigest()
    return {
        "commit": commit,
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        "status_porcelain": status.splitlines() if status else [],
        "diff_sha256": diff_sha,
        "source_verified": bool(commit) and dirty is False,
    }


def source_patch(root: str | Path) -> bytes | None:
    """Patch of a dirty tree against HEAD, for the run record (``git apply``-able).

    ``git diff HEAD --binary`` covers tracked files; untracked, non-ignored files are
    appended as new-file diffs, so the recorded patch reproduces the whole working tree.
    """
    patch = _git(root, "diff", "HEAD", "--binary", text=False)
    untracked = _git(root, "ls-files", "--others", "--exclude-standard") or ""
    parts = [patch] if patch else []
    for name in sorted(filter(None, untracked.splitlines())):
        # --no-index exits 1 when the files differ, which _git treats as a failure;
        # run it directly instead
        try:
            proc = subprocess.run(["git", "diff", "--binary", "--no-index", "--", "/dev/null",
                                   name], cwd=root, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode in (0, 1) and proc.stdout:
            parts.append(proc.stdout)
    return b"".join(parts) if parts else patch


def same_source(a: dict | None, b: dict | None) -> bool:
    """True when two ``git_info`` records describe the same code (commit + local edits)."""
    if not a or not b:
        return False
    return (a.get("commit") is not None and a.get("commit") == b.get("commit")
            and bool(a.get("dirty")) == bool(b.get("dirty"))
            and a.get("diff_sha256") == b.get("diff_sha256"))


def package_versions() -> dict[str, str | None]:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def environment_info(root: str | Path | None = None) -> dict:
    versions = package_versions()
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
