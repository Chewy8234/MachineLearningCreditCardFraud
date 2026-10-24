# Running the experiments on NRP (Nautilus), CPU only

Status: **drafted, not yet run on the cluster.** Nothing here has been built, deployed or measured;
every number marked *provisional* must be replaced by a measurement. Rules below are taken from the
NRP docs (fetched 2026-10-24: *CPU-only jobs*, *Jobs tutorial*, *Cluster policies*, *Ceph storage*,
*Getting started*, *Building in GitLab*). Re-check them; policies change.

## Why this architecture

One **Indexed Job**, one pod per *(seed, scenario)* unit (12 units for `paper.yaml`). Inside a pod the
existing runner and the Flower/Ray simulation run unchanged, so the three banks stay in one pod; spreading
simulated clients over pods would add network and failure modes without speeding up Random Forest.
Kubernetes does not accelerate a single training process; it runs independent units side by side.

```mermaid
flowchart LR
  L[Laptop: kubectl + kubelogin] -->|apply| J
  GL[NRP GitLab: Kaniko build] -->|image| REG[(gitlab-registry.nrp-nautilus.io)]
  subgraph NS[namespace fgcu-vinodahuja-aml]
    J[Indexed Job: 12 completions, parallelism P] --> P0[Pod idx 0: runner seed42/iid]
    J --> P1[Pod idx 1: runner seed42/moderate]
    J --> PN[Pod idx ...]
    P0 & P1 & PN --> RES[(CephFS PVC, RWX)<br/>/results/TAG/unit_NN_*<br/>unique dir per pod]
    RAW[(PVC /raw/creditcard.csv<br/>read-only)] --> P0 & P1 & PN
    P0 -.local cache.-> S0[emptyDir /scratch<br/>splits + partitions rebuilt per pod]
  end
  REG --> J
  RES -->|kubectl cp via loader pod| L
  L --> M[merge_units.py -> generate_report.py]
```

Unit index -> `(seed, scenario)` is fixed by the config order (`python deploy/run_unit.py --list`).
Within a unit the runner executes all methods and the 6 ablations serially, so `local` exists for the KD
round-0 diagnostic exactly as in the laptop run.

### Concurrency safety (verified against the code)

| Shared thing in the original runner | How it is avoided |
|---|---|
| `data/processed`, `data/partitions` caches (unlocked) | per-pod `emptyDir` via `data.processed_dir` / `data.partitions_dir` overrides; rebuilt deterministically from the raw CSV |
| `environment.json`, `config.yaml`, `*_audit/split_manifest.json`, `runner.log`, `ledger.jsonl` at the results root | `results_dir` override: one root per unit |
| run directories | already unique per (seed, scenario, method/ablation) |
| duplicate execution of an index | Indexed Job guarantees one pod per index at a time; resume skips `completed` runs and archives interrupted ones |

No file under `src/` is modified. Rebuilding the cache per pod must be **checked**: compare
`proxy_row_id_sha256` / split hashes in `split_manifest.json` and `partition_manifest.json` with the laptop's.

## Resource sizing (provisional)

NRP rules that bind us: limits ≤ 1.2 × requests (use equal, Guaranteed QoS; required above ~100 pods);
a user may not have more than 4 pods outside CPU use of 20–200 % of requested, RAM 20–150 % of requested
(1 CPU / 2 GB pods exempt); don't over-request. RF uses all cores during a fit, XGBoost/MLP/partitioning/
evaluation do not, so **average** utilisation of a 16-core pod may fall below 20 %. Hence measure first.

| Candidate | Expectation (not measured) |
|---|---|
| 4 cores/bank (original) | baseline; the validation job uses this |
| 8, 12, 16 | RF speedup is sublinear; memory-bandwidth and the serial parts (soft-target expansion, joblib tree-merge) flatten it. Bench decides. |

RAM 24 Gi is a placeholder. Measure peak RSS (`bench_threads.py` prints it; `kubectl top pod` while running)
and set requests = limits ≈ 1.3 × peak.

**Important:** with `CLIENT_NUM_CPUS=N` both RF *and XGBoost* use `n_jobs=N` (the same value is the Ray
reservation). A dedicated `federation.rf_n_jobs` option would isolate RF, but it touches `src/`; it is only worth
adding after the benchmark shows XGBoost's thread count matters. Ray is given `3×N` *logical* CPUs on a pod that
physically has N; only RF is busy for long, so this is not oversubscribed in practice, but verify with `top`.

## Research-integrity rules for this deployment

1. **Platform change.** Laptop = macOS arm64; the image is Linux x86_64. Floating-point results can differ in
   the last digits across architectures and thread counts. Do **not** expect bit-identical outputs: compare with
   `deploy/compare_runs.py` and judge differences against the bootstrap CIs.
2. **Never mix tags.** Each run set has a `RUN_TAG` (hardware setting). `hardware.json` records CPU model, cgroup
   limit, thread env, node, pod, image. Do not put runtimes from different tags in one comparison table.
3. **Where each result set may be used.** Main-paper results stay the laptop run (`results/full`, complete, 111/111).
   NRP is for new sets (random-proxy sensitivity, extra seeds/datasets), or a *full* NRP re-run reported
   as its own, separately labelled platform. A table that mixes laptop and NRP *cells* is not defensible.
4. Seeds, partitions, hyperparameters and metrics are untouched; only output paths and `client_num_cpus` change.
5. Thread-count effect on outputs is **unknown until `bench_threads.py` is run**. Run it before trusting a changed setting.

## Step-by-step

Replace `<GROUP>/<PROJECT>` with your NRP GitLab project. `NS=fgcu-vinodahuja-aml`.

**0. Confirm you may do this.** Accept the NRP AUP (linked from *Cluster policies*) and ask the namespace
admin whether this research use and the CPU volume are within the allocation. Not verifiable from here.

**1. Authenticate and install tools (macOS).**
```bash
brew install kubectl                       # official alternative: kubernetes.io install docs
# kubelogin plugin is mandatory ("your kubeconfig will not work" otherwise); NRP's page has the full script:
# download the kubelogin release for macOS and: sudo mv ./kubelogin /usr/local/bin/kubectl-oidc_login
```

**2. Add the NRP kubeconfig without overwriting your own.** NRP's one-liner `curl -o ~/.kube/config` *replaces*
the file. Safer:
```bash
cp ~/.kube/config ~/.kube/config.bak 2>/dev/null     # back up whatever exists
mkdir -p ~/.kube && curl -fSL -o ~/.kube/nrp-config "https://nrp.ai/config"
export KUBECONFIG=~/.kube/nrp-config                 # only in this terminal; your other clusters stay untouched
kubectl config use-context nautilus
kubectl config set contexts.nautilus.namespace fgcu-vinodahuja-aml
kubectl get pods                                     # first call opens the CILogon browser login
```
(NRP's page says `--kubeconfig` does not work for its config; the env-var approach is not documented there, so if
login fails, fall back to the backup-and-replace method.)

**3. Verify permissions and quota.**
```bash
kubectl auth can-i create jobs ; kubectl auth can-i create persistentvolumeclaims
kubectl describe quota                  # CPU/memory/storage caps for the namespace (may be empty)
kubectl get storageclass                # expect rook-cephfs (RWX), rook-ceph-block (RWO)
kubectl version                         # podFailurePolicy needs a recent server; see job-indexed.yaml note
```

**4. Create storage.** `kubectl apply -f deploy/k8s/pvc.yaml && kubectl get pvc`  (status `Bound`).

**5. Build and publish the image.** Create a project on `gitlab.nrp-nautilus.io`, push this repo's tracked
code (no `data/`, no `results/`, no kubeconfig; `.gitignore` already excludes data/results), add
`deploy/.gitlab-ci.yml` as `.gitlab-ci.yml` at the project root, and let the pipeline run (Kaniko). The image
appears under *Deploy → Container Registry*. A private project needs a deploy token and an `imagePullSecret`
(see NRP *Private Repos*); simplest is a public project, which is fine because the image contains no data.
Put the resulting image reference into both Job files (`REPLACE_*`).

**6. Upload the raw dataset once.** The CSV is licensed via Kaggle; it is not in the image.
```bash
kubectl apply -f deploy/k8s/pod-loader.yaml && kubectl wait --for=condition=Ready pod/heterofl-loader
kubectl exec heterofl-loader -- mkdir -p /store/raw /store/results
kubectl cp data/raw/creditcard.csv heterofl-loader:/store/raw/creditcard.csv
kubectl exec heterofl-loader -- sha256sum /store/raw/creditcard.csv   # must match the README's 76274b69…98551a89
```
Keep the loader pod only while copying: `kubectl delete pod heterofl-loader`.

**7. Validate ONE unit (original setting).**
```bash
kubectl apply -f deploy/k8s/job-validate.yaml
kubectl get pods -w                       # Pending -> Running -> Completed
kubectl logs -f job/heterofl-validate
```
Then fetch and compare (step 10) against `results/full` seed 42 / iid. Proceed only if differences are
negligible relative to the bootstrap CIs and the manifests' hashes match.

**8. Benchmark thread counts (before scaling).** Locally, on an idle machine (or in a pod): see *Benchmarking*.

**9. Run the full set.** Edit `RUN_TAG`, `CLIENT_NUM_CPUS`, resources and `parallelism` in `job-indexed.yaml`,
then start with a small `parallelism` (2–4):
```bash
kubectl apply -f deploy/k8s/job-indexed.yaml
kubectl get job heterofl-paper-cpu12 ; kubectl get pods -l job-name=heterofl-paper-cpu12
kubectl top pod                           # CPU/RAM actually used vs requested (NRP checks this)
kubectl logs <pod-name> --tail=50 ; kubectl describe pod <pod-name>
```
Raise `parallelism` only if utilisation is in NRP's accepted ranges and the quota allows.

**10. Retrieve results.**
```bash
kubectl apply -f deploy/k8s/pod-loader.yaml && kubectl wait --for=condition=Ready pod/heterofl-loader
kubectl cp heterofl-loader:/store/results/nrp-paper-cpu12-v1 results/nrp/nrp-paper-cpu12-v1
kubectl delete pod heterofl-loader
python deploy/compare_runs.py results/full results/nrp/nrp-validate-cpu4/unit_00_seed42_iid
python deploy/merge_units.py results/nrp/nrp-paper-cpu12-v1 --out results/full_nrp_cpu12
python scripts/generate_report.py --results results/full_nrp_cpu12
```

**11. Resume.** Failed pods are retried (`backoffLimit`); preemptions are ignored by `podFailurePolicy`. If the
whole Job fails or you deleted it, re-apply the same file: units write to the same `RUN_TAG` directory,
completed runs are skipped, interrupted ones are archived under `_archive/` and redone.

**12. Clean up (ask before deleting results).**
```bash
kubectl delete job heterofl-validate heterofl-paper-cpu12      # removes the pods; results stay on the PVC
kubectl get pods,jobs                                          # confirm nothing is left behind (NRP asks for this)
# only after you copied and verified the results:  kubectl delete pvc heterofl-data
```
Jobs auto-delete after `ttlSecondsAfterFinished` (3 days here; NRP default is 7). Never run `sleep infinity` in a Job.

## Benchmarking plan

Measured vs. estimated must be kept apart in the write-up.

1. **Thread effect + determinism** (`deploy/bench_threads.py`, idle machine, same data cache):
   `python deploy/bench_threads.py --threads 4 1 8 12 16`. Output: fit time, peak RSS, `max |Δp|` and bitwise
   equality vs. the reference thread count, for RF and XGBoost, local and KD-style fits.
2. **NRP validation unit**: wall time, per-round `fit_seconds` (in `rounds.csv`), `kubectl top` CPU/RAM, queue time
   (Job creation → pod Running), total CPU-hours = Σ(requests × pod runtime).
3. **Full set per setting** (4/8/12/16 cores): total wall time, mean per unit, RF time share, peak RAM, CPU-hours,
   and all evaluation metrics compared to the laptop reference with `compare_runs.py`.
4. Report speedup as a *measured* ratio per setting; timing tables use one tag at a time.

## Reproducibility and validation checklist

- [ ] AUP accepted; namespace admin confirmed research use
- [ ] Raw CSV SHA-256 on the PVC matches the README
- [ ] Image built from a clean commit; digest and commit recorded (`IMAGE_REF`, `environment.json`)
- [ ] Split/partition manifest hashes in the pod equal the laptop's
- [ ] `bench_threads.py` run; thread-count effect on probabilities documented
- [ ] Validation unit compared with `compare_runs.py`; differences ≪ bootstrap CI
- [ ] Each result set has one `RUN_TAG`; `hardware.json` present for every unit
- [ ] 12/12 units complete; ledger shows no failed runs; no duplicate (seed, scenario) at merge
- [ ] Timing tables use a single hardware setting
- [ ] No kubeconfig/token in repo or image; PVC and pods cleaned up
