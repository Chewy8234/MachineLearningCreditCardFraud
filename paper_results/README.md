# Paper results

Reports, tables (LaTeX + CSV), figures and provenance of the experiment runs, copied
from `results/<profile>/` (which is git-ignored). Trained models and per-transaction
predictions are not included; rerun the profile to regenerate them.

| Folder | Profile | Runs | Contents |
|---|---|---|---|
| `full/` | `configs/paper.yaml` | 111/111 completed | local-only, centralized, FedAvg MLP, ensemble (no KD), heterogeneous KD committee + ablations; 3 seeds x 4 scenarios |
| `homog_xgboost/` | `configs/homog_xgboost.yaml` | 48/48 completed | federated XGBoost (bagging), all-XGBoost KD committee and ensemble |

Run on Windows 11, CPython 3.11.0, Flower 1.26.1, xgboost 3.2.0, torch 2.14.1,
scikit-learn 1.9.1, from commit `3a0f650` (clean tree; provenance verified for every
run). Dataset: ULB `creditcard.csv`, SHA-256 `76274b69...98551a89`.

Start with `full/reports/summary.md`.
