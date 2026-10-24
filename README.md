# Heterogeneous Federated Committees for Credit Card Fraud Detection

Can three simulated banks that use **different model families** — XGBoost,
Random Forest and a PyTorch MLP — help each other detect fraud by exchanging
**predictions on a shared public reference set** (federated knowledge
distillation), and how does that compare with homogeneous **FedAvg**, with
isolated local models, and with simply averaging the heterogeneous models'
predictions?

This repository is a complete, reproducible experiment on the ULB Credit Card
Fraud Detection dataset, built on **Flower 1.42** (genuine ServerApp/ClientApp
message exchange in the Flower Simulation Runtime). The hypothesis is tested,
not assumed; negative results are reported as observed.

* Methodology: [docs/methodology.md](docs/methodology.md)
* Architecture: [docs/architecture.md](docs/architecture.md)
* Experiment protocol and output layout: [docs/experiment_protocol.md](docs/experiment_protocol.md)
* Limitations: [docs/limitations.md](docs/limitations.md)
* Reproducibility, provenance and safe resume: [docs/reproducibility.md](docs/reproducibility.md)

## Experimental conditions

| # | Condition | Federated? | Systems evaluated |
|---|---|---|---|
| 1 | Local-only | no | `bank_A` (XGBoost), `bank_B` (RF), `bank_C` (MLP) |
| 2 | Centralized **reference** (pooled client data) | no — hypothetical | `centralized_{xgboost,random_forest,mlp}` |
| 3 | Homogeneous FedAvg (3 × MLP, example-weighted) | yes (Flower) | `fedavg_global` |
| 4 | Heterogeneous ensemble, no KD (equal-weight average of the local models) | no | `ensemble_equal` |
| 5 | **Heterogeneous KD** (proposed): leave-one-out teachers on the proxy set, model-specific soft-target training, 5 rounds | yes (Flower, custom strategy) | `kd_bank_{A,B,C}`, `kd_committee` |

Ablations: λ_KD ∈ {0, 0.1, 1.0} (primary 0.5), full-committee vs
leave-one-out teachers, validation-AP-weighted vs equal teacher/ensemble
weights, an extra heterogeneity level (α = 1.0), and a naturally sampled
proxy set (`configs/paper_random_proxy.yaml`).

## Installation

Tested with CPython 3.12.14 on macOS arm64 (12-core CPU, no GPU).
Several pinned dependencies (xgboost 3.4.1, numpy 2.5.4, scipy 1.18.1)
require Python ≥ 3.12. POSIX only (macOS/Linux): the federation launcher
kills timed-out simulations by process group (`os.killpg`) and the results
directory lock uses `fcntl`.

```bash
uv venv --python 3.12 .venv            # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"              # or: pip install -e ".[dev]"
# exact tested environment instead: uv pip install -r requirements-lock.txt && uv pip install -e . --no-deps
```

`requirements.txt` pins the direct dependencies; `requirements-lock.txt`
is the full transitive lock of the tested environment.

### Dataset

Place the Kaggle file at `data/raw/creditcard.csv`
(<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>), e.g. with
kagglehub:

```bash
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('mlg-ulb/creditcardfraud'); shutil.copy(p + '/creditcard.csv', 'data/raw/creditcard.csv')"
```

The loader verifies the schema, values and labels, writes an audit report,
and records the file's SHA-256 (canonical release:
`76274b69…98551a89`). Without the file, every command prints these
instructions. A synthetic, clearly **non-publishable** stand-in exists for
pipeline testing: `python -m heterofl.data.synthetic` and
`configs/smoke_synthetic.yaml`.

## Running

```bash
.venv/bin/python -m pytest -q                 # all tests incl. Flower/Ray simulations, ~2.5 min
.venv/bin/python -m pytest -q -m "not slow"   # without Ray/Flower, ~15 s
.venv/bin/python -m pytest -q -m slow         # only the Flower simulations + end-to-end runs

# PROFILE A, smoke (~2 min). results/smoke holds legacy results, so use a new directory:
.venv/bin/python -m heterofl.experiments.runner --config configs/smoke.yaml \
    --set results_dir=results/smoke_v2
# PROFILE B, full paper matrix (~75 min on 12 cores), from a clean commit:
.venv/bin/python -m heterofl.experiments.runner --config configs/paper.yaml \
    --require-clean-source --set results_dir=results/full_v2
.venv/bin/python scripts/generate_report.py --results results/full_v2   # tables, figures, statistics
```

`scripts/run_smoke.sh` / `scripts/run_all.sh` wrap the same commands (the latter
also runs the random-proxy sensitivity profile and both reports). Their default
output directories hold legacy results, so set new ones:
`SMOKE_RESULTS=results/smoke_v2 scripts/run_smoke.sh`,
`FULL_RESULTS=results/full_v2 PROXY_RESULTS=results/full_random_proxy_v2 scripts/run_all.sh --require-clean-source`.

* **Safe resume.** Re-running the same command skips completed runs, but only
  after verifying that each run's recorded identity (complete effective
  configuration, dataset SHA-256, library versions, partition, seed) and source
  revision match. A changed configuration is refused (exit status 4, listing the
  differing keys) instead of silently reusing or mixing results: use a new
  `results_dir`. Failed or interrupted runs are archived to `_archive/` with
  their logs and re-run. `--overwrite` re-runs completed runs (archiving, never
  deleting, the old ones). See [docs/reproducibility.md](docs/reproducibility.md).
* **Strict configuration.** Unknown keys (`--set kd.lamda_kd=1`,
  `models.xgboost.n_estimator`) and out-of-range values are errors.
* Subsets and overrides: `--seeds 42 --scenarios iid --methods local ensemble`,
  `--set kd.lambda_kd=1.0`, `--no-ablations`, `--ablations-only`,
  `--allow-source-change`, `--require-clean-source`.
* Smoke outputs carry a "NOT FOR PUBLICATION" banner, and so does any result
  produced from an uncommitted working tree.

### Debugging with Flower directly

Every federated job is a `job.json`; it can be executed outside the runner:

```bash
python -m heterofl.experiments.runner --config configs/smoke.yaml --prepare-job kd   # prints the job path
python -m heterofl.federation.simulation --job <job.json>                            # Flower Simulation Runtime
```

With the Flower CLI (`flwr run`, using `[tool.flwr.app]` in `pyproject.toml`):

```bash
scripts/flwr_debug.sh kd        # or fedavg; wraps:
flwr run . --stream --run-config 'job-path="<job.json>"' --federation-config "num-supernodes=3"
```

### Troubleshooting

* `ModuleNotFoundError: No module named 'heterofl'` although it is
  installed: Python ≥ 3.12 skips `.pth` files carrying the macOS `hidden`
  flag, and on this machine something repeatedly set that flag on every file
  in `.venv`. Fix with `chflags -R nohidden .venv`, or run with
  `PYTHONPATH=src`. The `scripts/*.sh`, `pytest`, `generate_report.py` and the
  federation launcher already add `src/` themselves.
* A crash or hang when XGBoost and PyTorch share a process (macOS): always
  import `heterofl` before `torch`; see docs/architecture.md.

`flwr run` (and Ray) cannot start workers from an interpreter whose path
contains whitespace, so it needs a venv at a space-free location
(e.g. `~/.venvs/heterofl`); the script checks this. The experiment runner
works around the limitation automatically. `flwr run` leaves a local
SuperLink daemon running (`pkill -f flower-superlink`).

## Repository layout

```
configs/      paper.yaml smoke.yaml models.yaml paper_random_proxy.yaml smoke_synthetic.yaml
src/heterofl/
  data/          loader, validation (audit), preprocessing, partitioning, pipeline, synthetic
  models/        xgboost_model, random_forest, mlp, model_factory, base
  distillation/  aggregation (teachers), soft_targets (weighted expansion), trainers
  federation/    client_app, server_app, fedavg (StrictFedAvg), heterogeneous_strategy,
                 messages, state, job, simulation (launcher)
  experiments/   local, centralized, homogeneous (FedAvg), ensemble, heterogeneous (KD),
                 federated_common, common, runner
  evaluation/    metrics, statistics, visualization, latex_tables, report
  experiments/identity  experiment/run identity, resume checks, results lock
  utils/         config (schema, defaults, hashing), logging, io, reproducibility (seeds, git provenance)
scripts/      run_smoke.sh run_all.sh generate_report.py flwr_debug.sh legacy/
deploy/       Dockerfile, k8s/ templates, run_unit.py, merge_units.py, check_manifests.py (NOT built/tested)
tests/        test_config test_resume test_data test_partitioning test_leakage test_models
              test_soft_targets test_aggregation test_metrics test_federation test_deploy test_end_to_end
docs/         architecture methodology experiment_protocol limitations reproducibility nrp_deployment
data/         raw/ processed/ partitions/        (generated, git-ignored)
results/      smoke/ full/ full_random_proxy/    (generated, git-ignored)
```

`scripts/legacy/kagglehub_load_example.py` (formerly `main.py`) is the earlier
kagglehub loading snippet; it is not used by the pipeline.

`deploy/` (Docker image, Kubernetes jobs for NRP) is a draft: the image has never
been built and no manifest has been applied. Fill in the placeholders and run
`python deploy/check_manifests.py deploy/k8s/job-*.yaml` before `kubectl apply`
([docs/nrp_deployment.md](docs/nrp_deployment.md)).

## Results

> **Provenance caveat.** These numbers come from the *legacy* result sets in
> `results/`, produced before configuration hashes and source diffs were
> recorded. They report `git commit 2c9adb7 (dirty)`, which does not identify
> the code; their configurations match `configs/` and the repaired code trains
> bit-identically to the committed pre-repair code, but the exact source cannot
> be verified. Regenerate them from a clean commit before publication
> ([docs/reproducibility.md](docs/reproducibility.md), "Legacy results").

Full paper profile, executed 2026-10-24 (111/111 runs completed, 0 failed;
74 min on a 12-core Apple Silicon laptop). Test AP, mean ± SD over seeds
42/123/2026, which share one test set (56,962 transactions, 98 frauds). Full
tables: `results/full/tables/`, figures: `results/full/figures/`, statistics and
caveats: `results/full/reports/summary.md`.

| System | IID | Mild (α=1.0) | Moderate (α=0.5) | Severe (α=0.1) |
|---|---|---|---|---|
| Local A (XGB) | 0.866 ± 0.010 | 0.834 ± 0.059 | 0.846 ± 0.064 | 0.825 ± 0.046 |
| Local B (RF) | 0.856 ± 0.016 | 0.839 ± 0.007 | 0.852 ± 0.001 | 0.784 ± 0.079 |
| Local C (MLP) | 0.798 ± 0.014 | 0.801 ± 0.011 | 0.843 ± 0.018 | 0.805 ± 0.009 |
| Central. XGB (ref.) | 0.886 ± 0.003 | — | 0.886 ± 0.003 | 0.886 ± 0.003 |
| Central. RF (ref.) | 0.877 ± 0.003 | — | 0.877 ± 0.003 | 0.877 ± 0.003 |
| Central. MLP (ref.) | 0.847 ± 0.027 | — | 0.847 ± 0.027 | 0.847 ± 0.027 |
| Ensemble, no KD (equal) | 0.878 ± 0.006 | 0.870 ± 0.019 | 0.881 ± 0.007 | 0.859 ± 0.023 |
| FedAvg MLP | 0.843 ± 0.034 | 0.846 ± 0.032 | 0.831 ± 0.035 | 0.835 ± 0.041 |
| KD A (XGB) | 0.871 ± 0.014 | 0.856 ± 0.014 | 0.846 ± 0.057 | 0.847 ± 0.022 |
| KD B (RF) | 0.855 ± 0.015 | 0.848 ± 0.005 | 0.863 ± 0.007 | 0.772 ± 0.092 |
| KD C (MLP) | 0.824 ± 0.021 | 0.824 ± 0.042 | 0.848 ± 0.025 | 0.840 ± 0.025 |
| KD committee | 0.883 ± 0.004 | 0.872 ± 0.006 | 0.884 ± 0.005 | 0.855 ± 0.027 |

What the numbers do and do not support (descriptive; 3 seeds on one test
set, no significance claims):

* **Heterogeneous committees vs FedAvg (main RQ).** Both the KD committee and
  the non-distilled ensemble have higher AP than FedAvg in 11 of 12
  seed × scenario pairs (+0.020 to +0.053 mean). For the KD committee, 7 of 12
  per-seed paired bootstrap intervals exclude zero (test-example resampling for
  fixed trained models; descriptive, not a significance test, and not
  independent across seeds). In both comparisons the
  exception is one severe-non-IID seed. FedAvg's validation AP was still
  rising at round 5 in the IID and severe scenarios (Figure 4), so a longer
  FedAvg budget could narrow the gap.
* **KD vs simply combining predictions (RQ3) — essentially no difference.**
  KD committee − ensemble: +0.004 (IID), +0.002 (mild), +0.004 (moderate),
  −0.004 (severe). 1 of 12 bootstrap intervals excludes zero. Almost all of
  the committee's gain over FedAvg comes from combining heterogeneous models,
  not from distillation. Under severe skew the λ = 0 committee (0.868)
  scores higher than the primary λ = 0.5 committee (0.855).
* **KD vs isolated local models (RQ1) — mixed and model-dependent.**
  * The MLP bank improves in 11 of 12 pairs (+0.005 to +0.036). The λ = 0
    control (same extra epochs, no teacher) is lower than λ = 0.5 in all three
    ablated scenarios, so the gain is attributable to distillation, not extra
    training.
  * XGBoost: +0.005 (IID), ≈ 0 (moderate), +0.022 (mild and severe, each
    with 2 of 3 seeds improved and a large spread).
  * Random Forest gets **worse** under severe non-IID (−0.012; 2 of 3 seeds
    lower).
* **Heterogeneity (RQ2).** Realised heterogeneity varies strongly between
  draws of the same α (Figure 1), including quantity skew; in one draw a
  single bank holds 96 % of the data. Relative to IID, the committees and
  the tree models lose AP under α = 0.1, while the MLP-based systems do not
  (the local MLP is slightly higher). No monotone trend in α is visible at
  this sample size.
* **Equal vs validation-weighted aggregation (RQ4).** Weighting the ensemble
  by validation AP changes its AP by ≤ 0.002; weighting the KD teachers by
  validation AP changes the committee's AP by ≤ 0.004.
* **Cost (RQ5).** Per run, KD exchanged 2.56 MB vs 0.52 MB for FedAvg (proxy
  vectors of 19,367 float32 values vs a 4,129-parameter MLP). KD federation
  wall time was 62 s vs 2 s, dominated by retraining the Random Forest every
  round.
* **Centralized reference.** Pooled XGBoost (0.886) is at or above every
  collaborative system. This is a hypothetical non-federated reference, not
  a bound.

**Sensitivity: naturally sampled proxy set** (`configs/paper_random_proxy.yaml`,
36/36 runs completed, `results/full_random_proxy/`). The uniformly drawn proxy
contains 35 frauds versus 34 in the stratified one, so this run mainly
measures sensitivity to *which* proxy rows are drawn (and hence which client
pool remains), not to a different class composition. The conclusions hold:

* both heterogeneous committees beat FedAvg (+0.013 to +0.044 AP);
* KD committee − ensemble is again small but **changes sign** relative to the
  primary run (+0.000 IID, −0.008 moderate, +0.011 severe), consistent with
  no reliable difference;
* mean per-bank KD gains are larger and positive for all three banks in all
  scenarios, including RF under severe skew (+0.019, versus −0.012 in the
  primary run). The per-bank effect of KD is therefore not robust to the
  proxy and partition draw.

## Key implementation decisions

* **Leakage control by construction.** Flower apps can only read a
  per-scenario *federation view*: each bank's private rows, proxy
  **features without labels**, and validation data. The test split is never
  written there; only the evaluation harness reads it, after model
  selection. Tests fail on any overlap.
* **One preprocessing for all**: log-Amount + standardisation fitted on the
  public proxy features only.
* **Fixed split, varying partitions**: test/validation/proxy are identical
  across seeds (paired comparisons); seeds vary partitions and model
  randomness.
* **KD round 0 = local-only models** (same seeds) — verified exact
  (max |Δp| = 0), so before/after comparisons isolate distillation.
* **Exact soft targets for XGBoost**, a documented approximation for Random
  Forest (weighted bootstrap with `max_samples=1.0`; λ = 0 reproduces the
  local forest exactly), normalised `L_sup + λ·L_KD` for the MLP.
* **Strict federation**: all three banks must reply in every round;
  identities, proxy hashes and probability vectors are validated; failures
  are recorded, never silently averaged over.
