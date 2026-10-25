"""Refuse to apply Kubernetes manifests that still contain placeholder values.

    python deploy/check_manifests.py deploy/k8s/job-validate.yaml && kubectl apply -f deploy/k8s/job-validate.yaml

The manifests in deploy/k8s/ are templates: the image reference (REPLACE_GROUP /
REPLACE_PROJECT / REPLACE_TAG) must be filled in before use. Exit status 1 lists
every placeholder found, so `&&` stops `kubectl apply`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"REPLACE_[A-Z_]+|<[A-Z_]+>|CHANGE_ME|TODO")


def find_placeholders(path: Path) -> list[tuple[int, str]]:
    hits = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]  # placeholders inside comments are documentation
        for m in PLACEHOLDER.finditer(code):
            hits.append((i, m.group(0)))
    return hits


def main(argv: list[str] | None = None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print("usage: check_manifests.py MANIFEST.yaml [...]", file=sys.stderr)
        return 2
    bad = 0
    for p in paths:
        for line, token in find_placeholders(p):
            print(f"{p}:{line}: placeholder {token!r} must be replaced before applying",
                  file=sys.stderr)
            bad += 1
    if bad:
        return 1
    print(f"OK: no placeholders in {len(paths)} manifest(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
