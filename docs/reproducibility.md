# Reproducibility, provenance and safe resume

This page describes what is recorded with every result, when a stored result may be
reused, and how to tell results of the current code apart from the legacy results that
were produced before these records existed.

## What every run records

| File | Where | Content |
|---|---|---|
| `experiment.json` | results root | experiment identity: semantic configuration, dataset SHA-256, library versions; `experiment_hash`, `config_hash`; source revision at creation |
| `config.yaml`, `environment.json`, `split_manifest.json`, `dataset_audit.json` | results root | written once, when the root is created |
| `invocations/<id>/` | results root | per invocation: `config.yaml` (incl. command-line overrides), `environment.json`, and `source.patch` (the `git diff HEAD`) if the tree was dirty |
| `ledger.jsonl` | results root | append-only: start / completed / failed per run, with `run_hash` |
| `status.json` | run dir | `status`, `run_id`, `run_hash`, `config_hash`, `experiment_hash`, `source_commit`, `source_dirty`, start/finish (Unix time and ISO-8601 UTC), wall time, error + traceback on failure |
| `run_identity.json` | run dir | the complete run identity (below), its hash, and the full source record |
| `config.yaml` | run dir | the effective configuration snapshot of this run (ablation overrides merged) |
| `manifest.json` | run dir | `provenance` (hashes, source, `source_verified`, `publication_reproducible`), partition and split manifests, environment |

The source record (`heterofl.utils.reproducibility.git_info`) holds the commit, branch,
`dirty`, the `git status --porcelain` lines and `diff_sha256` — a SHA-256 over
`git diff HEAD --binary` plus the names and contents of untracked files — so two dirty
trees on the same commit are still told apart. `source_verified` is true only when the
working tree is exactly a commit.

A run is labelled **`publication_reproducible`** only if the profile is `publishable`,
the data are not synthetic, **and** `source_verified` is true. The report applies the
same rule to the whole results directory and prints "NOT FOR PUBLICATION" with the
reasons otherwise. `--require-clean-source` makes the runner refuse to start on a dirty
tree.

## Configuration hashing

`heterofl.utils.config.config_hash` is the SHA-256 of a canonical JSON serialisation
(sorted keys, no whitespace, numpy scalars converted, types preserved — `1` and `1.0`
differ because e.g. scikit-learn treats `max_features=1` and `1.0` differently) of the
*semantic* configuration:

* the profile YAML with its `base_config` chain resolved,
* the models file (`models_config`) merged with the profile's `models:` section,
* all command-line `--set` overrides,
* every default the code would otherwise apply implicitly (`DEFAULTS` /
  `MODEL_DEFAULTS` in `config.py`), written into the configuration on load, so a value
  spelled out in YAML and the same value left implicit hash identically.

Excluded, because they cannot change any output:

| Excluded | Why |
|---|---|
| `results_dir`, `data.processed_dir`, `data.partitions_dir`, `project_root`, `config_path`, `models_config` | locations; the models file's *content* is included |
| `data.path` | location; the dataset's SHA-256 is part of the identity instead |
| `federation.round_timeout`, `federation.job_timeout` | can only make a run fail |
| `report`, `evaluation.bootstrap` | used only by the report generator, never by a run |
| `debug` (fault injection) | can only make a run fail, never alter a completed one |
| `seeds`, `methods`, `scenarios`, `ablations` | only *select* runs; each run's own seed, scenario definition, method, variant and merged ablation overrides are in its run identity |

Every key is validated against a schema (`SECTION_KEYS`, the model classes'
`PARAM_SPECS`); a typo such as `kd.lamda_kd` or `models.xgboost.n_estimator` is an
error, so it can never be silently ignored or hashed as a harmless extra key.

## Run identity

`run_identity` = semantic configuration of the run (ablation overrides merged),
`config_hash`, dataset SHA-256, versions of the numerical libraries (flwr, ray,
xgboost, torch, scikit-learn, pandas, numpy, scipy, ...), seed, the scenario's
definition (type, α; the `methods` filter and the `ablation` label are excluded),
method, variant, and the fingerprint of the realised partition (row-id hashes and class
counts of every set). `run_hash` = SHA-256 of that record. `run_id`
(`<profile>-<UTC time>-<random>/<seed>/<scenario>/<variant>/<method>`) names one
*execution*; `run_hash` names *what* was executed.

## Resume policy (one predictable rule: refuse, never mix)

1. A results directory belongs to exactly one experiment identity. If `experiment.json`
   exists with a different `experiment_hash`, the runner exits with status 4 and names
   the differing keys **before writing anything** (only an audit line is appended to
   `runner.log`). Use a new `results_dir` for a new configuration.
2. Before any run starts, every planned run is checked (pre-flight). A run whose
   `status.json` says `completed` is skipped only if its `run_identity.json` hash equals
   the requested `run_hash`, its outputs are present, and it was produced by the same
   source revision (commit + dirty diff hash). Otherwise the invocation stops with
   `ConfigMismatchError` / `SourceMismatchError`; nothing has been executed.
3. `--allow-source-change` reuses runs produced by another revision; they are recorded as
   `skipped-completed-other-source` in `outcomes_<invocation>.csv`, and the report then
   lists more than one source revision and withholds the publication label.
4. Failed or interrupted runs (`status` ≠ `completed`) contain no results; they are
   archived to `_archive/<run>-stale-<status>-<time>/` and re-run. `--overwrite` also
   archives completed runs and re-runs them. Nothing is deleted.
5. The ensemble re-uses the local-only models and checks that the local run's identity
   matches its own in everything the local models depend on (all sections except
   `ensemble` and `kd`); the KD round-0 determinism diagnostic does the same and is
   skipped (with the reason recorded) when the check fails.
6. An exclusive `flock` on `<results>/.heterofl.lock` prevents two processes from
   writing one results tree; the kernel releases it if the process dies. Configuration
   and JSON files are written atomically (temporary file + rename).

## Legacy results (produced before identity recording)

`results/full`, `results/full_random_proxy`, `results/smoke` and
`results/smoke_synthetic` were produced on 2026-10-24 between 20:02 and 22:12 UTC. Their
`environment.json` records `git commit 2c9adb7 (dirty: True)`: at that time almost all
of the source was untracked, so the commit does not identify the code. The source was
first committed as `626c8db` (tagged `pre-repair-baseline`) at 22:11:51 UTC, i.e. after
the runs. What can be said:

* Their `config.yaml` files are identical to what `configs/*.yaml` at `626c8db` load to
  (checked key by key), and their dataset SHA-256 is the canonical ULB file.
* It **cannot** be verified that the code that produced them is `626c8db`: there is no
  diff hash or patch from that time.
* The repaired code trains bit-identically to `626c8db` (verified on all three model
  families, local fit and two distillation rounds at λ ∈ {0, 0.5, 1}; and by re-running
  the smoke profile, see the repair report), so `626c8db` *is* consistent with them;
  consistency is not proof.

They are kept unmodified. They have no `experiment.json` and no `run_identity.json`, so
the runner refuses to resume into or extend them, and regenerating their report would
mark them as legacy. **To tell them apart:** a result produced by the repaired code has
`experiment.json` at its root and `run_identity.json` in every run directory, and its
`reports/summary.md` shows the experiment identity and source revision(s). For
publication, re-run the paper profile from a clean commit into a new directory, e.g.

```bash
python -m heterofl.experiments.runner --config configs/paper.yaml \
    --require-clean-source --set results_dir=results/full_v2
```

## Random seeds

* `split.seed` fixes the test/validation/proxy split (identical for all run seeds).
* Each run seed controls the client partition (`derive_seed(seed, ...)`, SHA-256 based,
  stable across processes) and every model: `model_seed(seed, role, model_type)`. The
  KD round-0 model of a bank uses the same seed as its local-only model.
* The runner calls `set_global_seed(seed)` (Python, NumPy, torch) at the start of each
  run.
* The MLP draws its initial weights and dropout masks from torch's global RNG (PyTorch
  offers no generator argument for either). Both happen inside
  `torch.random.fork_rng`, seeded from `derive_seed(seed, "mlp-init")` and
  `derive_seed(seed, "mlp-dropout", call)`; shuffling uses a private `torch.Generator`.
  Constructing, training, evaluating or loading an MLP therefore leaves the caller's
  RNG state untouched, and results do not depend on what ran before.
* torch's intra-op thread count is set to 1 once, when `heterofl.models.mlp` is first
  imported, and re-applied from `models.mlp.torch_threads` by the runner and the Flower
  apps — never by a model constructor. (Multi-threaded torch next to XGBoost's OpenMP
  runtime deadlocks on macOS arm64.)

## Checkpoint trust

Checkpoints are loaded with `torch.load(weights_only=True)` (MLP: tensors and plain
containers only — a tampered file cannot run code; checkpoints written before this
change load unchanged), `XGBClassifier.load_model` (JSON data) and `joblib.load`
(Random Forest — **unpickling can execute arbitrary code**). The project therefore
loads only checkpoints it wrote itself:

* the Flower client loads its previous-round model only from the path it recorded in
  its own `context.state`, verified to lie inside its bank's state directory and to
  match the recorded SHA-256;
* the harness loads round models only by plain file names inside the run directory;
* a `job-path` received in a Flower message must be an absolute path to an existing
  `job.json`, inside `$HETEROFL_JOB_ROOT` if that variable is set.

These checks catch mistakes; they are not a security boundary. The simulation assumes a
trusted server, trusted clients and a trusted file system. It is **not** safe to point
these ClientApps at an untrusted SuperLink, or to load checkpoints from untrusted
parties, without replacing joblib/pickle serialisation and authenticating messages.

## Platform requirements

POSIX (macOS, Linux): the federation launcher uses `start_new_session` and
`os.killpg(..., SIGKILL)` to stop a timed-out simulation together with its Ray
workers, the results lock uses `fcntl.flock`, and a space-free symlink to the
interpreter is created for Ray. Windows is not supported.
