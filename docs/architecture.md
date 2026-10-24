# Architecture

## Components

```
configs/*.yaml ──► heterofl.experiments.runner ───────────────────────────────┐
                    │  (seed × scenario × condition matrix, ablations,        │
                    │   status/resume, ledger)                                │
                    ▼                                                         │
          heterofl.data.pipeline                                              │
          ├─ processed bundle   data/processed/<tag>/   (X, y, splits, scaler)│
          └─ scenario partition data/partitions/<tag>/seed_<s>/<scenario>/    │
                               └─ view/  ◄── the ONLY data a Flower app reads │
                                    bank_A.npz bank_B.npz bank_C.npz          │
                                    proxy.npz (features + ids, NO labels)     │
                                    validation.npz                            │
                    ▼                                                         │
   ┌──────────── experimental conditions (heterofl.experiments.*) ──────────┐ │
   │ local.py        centralized.py      ensemble.py    (in-process)        │ │
   │ homogeneous.py  heterogeneous.py ──► job.json ──► subprocess:          │ │
   │                                     python -m heterofl.federation.simulation
   │                                       └─ flwr.simulation.run_simulation │ │
   │                                            ServerApp (thread)           │ │
   │                                            3 × ClientApp (Ray actors)   │ │
   │ after training: model selection on validation → test evaluation        │ │
   └────────────────────────────────────────────────────────────────────────┘ │
                    ▼                                                         │
   results/<profile>/seed_<s>/<scenario>/<condition>/  ◄──────────────────────┘
                    ▼
   heterofl.evaluation.report  (scripts/generate_report.py)
   → tables/*.tex|csv, figures/*.png|pdf, reports/summary.md, bootstrap CSVs
```

## Process model

* The **runner** is a plain Python process. It prepares data, runs the three
  non-federated conditions in-process, and launches every federated job
  (FedAvg, KD, KD ablations) as a **separate interpreter**
  (`heterofl.federation.simulation`). Isolation means a crashed or hung
  simulation becomes a recorded failure (the child and its Ray processes are
  killed as a process group on `federation.job_timeout`), and Ray never shares
  a process with the harness.
* The child calls `flwr.simulation.run_simulation` (Flower 1.42 Simulation
  Runtime) with `num_supernodes = 3`. The ServerApp runs in a thread of the
  child; each SuperNode's ClientApp executes in Ray actors with
  `federation.client_num_cpus` CPUs.
* The same `ServerApp` / `ClientApp` objects are exposed for the Flower CLI
  (`[tool.flwr.app.components]` in `pyproject.toml`), so `flwr run` executes
  identical code (see README, "Debugging with `flwr run`").

## Flower messages

All exchanges use the Message API (`flwr.app.Message`, `RecordDict`,
`ArrayRecord`, `ConfigRecord`, `MetricRecord`).

| Message type | Direction | Content |
|---|---|---|
| `query.identify` | server → node | `config.job-path` |
| reply | node → server | `info`: partition-id, bank, model type, node id |
| `train` (FedAvg) | server → node | `arrays`: global MLP state dict; `config`: job-path, server-round |
| reply | node → server | `arrays`: updated MLP parameters; `metrics`: `num-examples` (FedAvg weight), train loss, local validation AP |
| `train.kd_init` | server → node | `config` only (round 0) |
| `train.kd_distill` | server → node | `teacher` (ArrayRecord, float32 vector over the ordered proxy set) + `config` (round, proxy hash, the bank the teacher is for) |
| KD reply | node → server | `proxy_predictions` (float32 vector), `metrics`, `info` (identity, proxy hash, model hash) |

KD replies must not contain parameters (the server rejects any reply that
carries an `arrays` record), and no KD message ever carries raw rows.

## Identity and state

* A SuperNode's `node_config["partition-id"]` (0, 1, 2) selects its bank
  and model type from `job.json`. The server first sends `query.identify`
  to every node and builds `node_id → bank`; every later reply must repeat
  the same identity or the round fails.
* KD clients persist their model after every round to
  `<job>/federation/clients/bank_<X>/round_<r>/` (native format: XGBoost JSON,
  joblib for RF, `torch.save` for the MLP) together with validation
  predictions, proxy predictions, the received teacher, and fit diagnostics.
  `context.state` holds a `ConfigRecord` with the round, path and SHA-256 of
  the latest checkpoint; a distillation round refuses to start if that record
  is missing, belongs to another bank, is from the wrong round, or the file
  hash differs (`ClientStateError`). This makes persistence independent of
  which Ray worker process happens to execute a node's message.

## Failure handling

| Failure | Behaviour |
|---|---|
| Client raises | reply carries an `Error`; server raises `FederationError` naming the bank and round |
| Missing reply / timeout | `FederationError` listing the missing banks (no silent partial aggregation; Flower's FedAvg default of averaging the survivors is overridden by `StrictFedAvg`) |
| Invalid probability vector (NaN, out of [0,1], wrong length) | `FederationError` |
| Proxy set/order mismatch (hash) | `FederationError` |
| Hung simulation | process group killed after `federation.job_timeout` |
| Any run failure | `status.json` = failed with traceback; logs kept; no metrics written; next invocation archives the run directory to `_archive/` and re-runs it |

## Platform notes (found while building this, macOS arm64)

* **OpenMP**: XGBoost and PyTorch load different `libomp` copies. Importing
  torch first makes `XGBClassifier.fit` segfault; after XGBoost has run,
  multi-threaded torch deadlocks. `heterofl/__init__.py` imports xgboost
  first, and the MLP uses one intra-op thread.
* **Paths with spaces**: Ray's raylet starts workers via a shell command
  built from `sys.executable`. The launcher runs the child interpreter
  through a space-free symlink to the venv (`$TMPDIR/heterofl-env-<hash>`).
  `flwr run` cannot be patched this way; it needs a space-free venv.
* **Hidden `.pth` files**: Python ≥ 3.12 skips `.pth` files with the macOS
  `hidden` flag (observed on the editable-install file in this repository's
  venv). The launcher also puts `src/` on `PYTHONPATH` for children; fix the
  venv with `chflags -R nohidden .venv`.
* **Telemetry**: Flower's anonymous telemetry is disabled for all launched
  jobs (`FLWR_TELEMETRY_ENABLED=0`).
