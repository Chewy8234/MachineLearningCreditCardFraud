"""Logging setup shared by the runner, the harness and the Flower apps."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"heterofl.{name}" if not name.startswith("heterofl") else name)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root ``heterofl`` logger to write to stderr (idempotent)."""
    root = logging.getLogger("heterofl")
    root.setLevel(level)
    if not any(getattr(h, "_heterofl_console", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(FORMAT))
        handler._heterofl_console = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.propagate = False


class FileLog:
    """Context manager that tees ``heterofl`` log records to a file."""

    def __init__(self, path: Path, level: int = logging.INFO):
        self.path = Path(path)
        self.level = level
        self.handler: logging.FileHandler | None = None

    def __enter__(self) -> "FileLog":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(self.path, mode="a")
        self.handler.setLevel(self.level)
        self.handler.setFormatter(logging.Formatter(FORMAT))
        logging.getLogger("heterofl").addHandler(self.handler)
        return self

    def __exit__(self, *exc) -> None:
        if self.handler is not None:
            logging.getLogger("heterofl").removeHandler(self.handler)
            self.handler.close()
