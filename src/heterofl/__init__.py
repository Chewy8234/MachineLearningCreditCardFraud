"""Heterogeneous federated committees for credit card fraud detection."""

# XGBoost and PyTorch load different OpenMP runtimes (libomp). On macOS arm64,
# loading torch's copy first makes XGBClassifier.fit segfault (verified with
# torch 2.14.1 / xgboost 3.4.1). This package __init__ runs before any heterofl
# submodule, so importing xgboost here fixes the load order for the runner, the
# federation subprocesses and the Ray workers. Import heterofl before torch in
# any external script that uses both.
import xgboost as _xgboost  # noqa: F401  (import-order side effect)

__version__ = "0.1.0"

#: Bank identifiers, in partition-id order.
BANKS = ("A", "B", "C")
