# Performance notes (validation campaign)

Machine: AMD Threadripper PRO 5955WX (16C/32T), 125 GB RAM, no GPU. Pinned stack: flwr 1.26.1, xgboost 3.2.0,
torch 2.14.1, scikit-learn 1.9.1, Python 3.12 (Linux).

## Where the time went (serial runner)
* Runs execute strictly serially; only 1-2 runner processes were active, leaving ~70% of the CPU idle.
* Inside a KD run each round is a synchronous barrier waiting for the Random Forest bank: RF fit ~17.7 s/round versus
  ~1 s (XGBoost) and 0.4-1.6 s (MLP), so two of three clients idle most of the round. Ray start-up adds ~10-25 s per federated run.

## Thread benchmark (`deploy/bench_threads.py`, moderate_non_iid, seed 42, idle machine)
| n_jobs | RF local fit | RF KD-style fit | max abs dp vs n_jobs=4 |
|---|---|---|---|
| 1 | 65.8 s | 129.8 s | 1.1e-16 |
| 4 | 18.2 s | 36.2 s | reference |
| 8 | 9.6 s | 18.8 s | 2.2e-16 |
| 16 | 5.3 s | 10.5 s | 1.1e-16 |
XGBoost predictions were bitwise identical for every thread count. Thread count is therefore numerically safe, but
`client_num_cpus=4` was kept for every run so that results stay directly comparable with the paper profile (and the
config hash is unchanged); concurrency was used to fill the machine instead.

## Parallel unit scheduler (`scripts/validation/run_units.py`)
One unit = one runner invocation for (group, seed, scenario) in its own results directory; queue run from a frozen
git worktree (single source commit, `source_dirty=false`), data cache pre-warmed serially, heaviest units first,
P=8 concurrent units, BLAS/OpenMP pinned to 1 thread.

**Equivalence gate (passed):** units re-run through the scheduler were compared with the earlier serial runs
(`scripts/validation/compare_predictions.py`, 22-51 prediction files each): 19-42 files bitwise identical, the
rest differ by <= 8e-7 in probability (float32 FedAvg) or <= 2.2e-16 (RF/committee), with test-AP difference 0.000000.

**Campaign:** 243 units (plus 3 gate units), 0 failures, 0 retries, 199 minutes wall-clock, CPU load ~31-34 of 32
throughout. Summed unit time was 25.9 h (each unit slowed by sharing the machine), i.e. ~7.8x concurrency. The
uncontended serial time was not measured; an estimate from early serial runs is 12-20 h.
