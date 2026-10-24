# Experiment protocol

Everything below is driven by configuration; no source edits are needed.

## Fixed before any test evaluation

* Split protocol and split seed (42), proxy construction (stratified), the
  preprocessing, partition constraints.
* Model hyperparameters (`configs/models.yaml`), round counts, local epochs,
  λ = 0.5, leave-one-out equal-weight teachers, equal-weight committee.
* Threshold rule (max F1 on validation), model selection (final round),
  primary metric (AP).
* The primary method. Ablations are reported alongside, not used to choose it.

## Profiles

| | Smoke (`configs/smoke.yaml`) | Paper (`configs/paper.yaml`) |
|---|---|---|
| Data | real ULB, 20 % stratified subsample | full ULB |
| Scenarios | moderate (α = 0.5) | IID, α = 1.0 (ablation E), 0.5, 0.1 |
| Seeds | 42 | 42, 123, 2026 |
| Rounds | 2 | 5 |
| Models | 60 trees XGB, 40 trees RF, MLP 32-16 | as in `models.yaml` |
| Ablations | λ = 0, full committee, val-weighted teacher, val-weighted ensemble | + λ = 0.1, 1.0 |
| Output | `results/smoke/` (banner: NOT FOR PUBLICATION) | `results/full/` |

`configs/paper_random_proxy.yaml` repeats the paper profile (without
ablations and the α = 1.0 level) with a uniformly sampled proxy set
→ `results/full_random_proxy/`. `configs/smoke_synthetic.yaml` runs the
smoke profile on synthetic data (no download needed; non-publishable).

## Run matrix (paper)

For each seed and scenario, in this order: `local`, `centralized`
(not for α = 1.0), `ensemble` (re-uses `local`), `fedavg`,
`heterogeneous_kd`; then, for IID / α = 0.5 / α = 0.1, the ablations
`kd_lambda_0.0`, `kd_lambda_0.1`, `kd_lambda_1.0`, `kd_full_committee`,
`kd_teacher_val_weighted`, `ensemble_val_weighted`. 111 runs in total.

## Commands

```bash
python -m heterofl.experiments.runner --config configs/paper.yaml          # run / resume
python -m heterofl.experiments.runner --config configs/paper.yaml \
       --seeds 123 --scenarios severe_non_iid --methods fedavg             # subset
python -m heterofl.experiments.runner --config configs/paper.yaml \
       --set kd.lambda_kd=1.0 --set results_dir=results/lambda1            # override (new dir!)
python scripts/generate_report.py --results results/full                   # tables/figures
```

A changed configuration always needs its own `results_dir`; the runner refuses
to mix it into an existing one.

## Resume, overwrite and failures

Full rules and rationale: [reproducibility.md](reproducibility.md).

* A results directory holds exactly one experiment identity
  (`experiment.json`). A configuration that differs in any semantic setting
  (λ, models file, any hyperparameter, aggregation, rounds, split seed,
  partition constraints, data subsample, library versions, ...) is refused with
  exit status 4 and a list of the differing keys; write it to a new
  `results_dir`. Changing only locations, timeouts or report settings resumes.
* Before anything runs, every planned run is checked. A run with
  `status.json: completed` is skipped only if its `run_identity.json` matches the
  requested run identity and it was produced by the same source revision
  (`--allow-source-change` relaxes the latter and records it).
* A failed or interrupted run directory is moved to
  `<scenario>/_archive/<run>-stale-<status>-<UTC time>/` and re-run.
* `--overwrite` re-runs completed runs; the previous directory is archived
  (`-overwritten-`), never deleted.
* Results directories without `experiment.json` (legacy) are never resumed or
  extended.
* `--require-clean-source` refuses to run on an uncommitted working tree.
* Partition failures are recorded in `<scenario>/partition_status.json`.
* `results/<profile>/ledger.jsonl` logs every start/finish/failure with its
  `run_hash`; `outcomes_<invocation>.csv` summarises each invocation.
* A lock (`.heterofl.lock`) stops a second process from writing the same
  results directory.

## Output layout

```
results/full/
  experiment.json config.yaml environment.json dataset_audit.json split_manifest.json
  ledger.jsonl runner.log outcomes_<invocation>.csv
  invocations/<invocation>/{config.yaml,environment.json,source.patch (dirty tree only)}
  seed_42/
    iid/
      partition_report.csv partition_manifest.json
      local/ centralized/ ensemble/ fedavg/ heterogeneous_kd/
        status.json run_identity.json config.yaml manifest.json run.log
        metrics.json metrics.csv runtime.json predictions/<system>.npz
        models/ (local, centralized)
        rounds.csv federation.log federation/ (fedavg, KD)
          job.json server/{status.json,rounds.json,server.log,global_round_XX/}
          clients/bank_X/round_XX/{model.*,meta.json,val_probs.npy,proxy_probs.npy,teacher.npy}
        diagnostics.json (KD: round-0 vs local-only, selected rounds)
      ablations/<name>/ (same structure)
  tables/ figures/ reports/
```

`manifest.json` in every run embeds the partition manifest (sizes, class
counts, row-id hashes of every set, overlap and coverage checks) and the
split manifest (dataset SHA-256, split seed, preprocessing provenance), plus
package versions, hardware and the provenance block (run/config/experiment
hashes, git commit, dirty flag and diff hash, `publication_reproducible`).
`run_identity.json` holds the complete identity the run was checked against.

The legacy directories `results/{full,full_random_proxy,smoke,smoke_synthetic}`
predate this layout (no `experiment.json`, no `run_identity.json`).

## Systems evaluated

| System id | Condition |
|---|---|
| `bank_A`, `bank_B`, `bank_C` | local-only XGBoost / RF / MLP |
| `centralized_xgboost`, `centralized_random_forest`, `centralized_mlp` | centralized reference |
| `fedavg_global` | FedAvg global MLP (final round) |
| `ensemble_equal` (`ensemble_validation_ap` in ablation D) | ensemble of local models |
| `kd_bank_A`, `kd_bank_B`, `kd_bank_C` | each bank's model after KD (final round) |
| `kd_committee` | equal-weight committee of the distilled models |
| `kd_bank_X_round0` | KD round-0 models (before distillation; diagnostic) |
