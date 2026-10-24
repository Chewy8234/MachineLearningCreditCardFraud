# Methodology

## Research questions → experimental conditions

| Question | Comparison |
|---|---|
| Main RQ: heterogeneous committee vs homogeneous FL under non-IID data | KD committee / KD bank models vs FedAvg MLP, per scenario |
| RQ1: does KD beat isolated local models? | `kd_bank_X` vs `bank_X` (seed-paired), Table 5 / Figure 7 |
| RQ2: effect of heterogeneity on KD vs FedAvg | all conditions across IID, α=1.0, α=0.5, α=0.1 (Table 4 / Figure 3), with the *realised* heterogeneity from Table 1 |
| RQ3: is KD more than combining predictions? | `kd_committee` vs `ensemble_equal` (identical members' architectures, identical combination rule; only training-time exchange differs) |
| RQ4: equal-weighted aggregation under different fraud distributions | ensemble/KD equal vs validation-AP-weighted (ablation D), ensemble vs best single member |
| RQ5: performance vs communication vs computation | Table 6 / Figure 8 |

## Data protocol

ULB `creditcard.csv` (284,807 rows, 492 frauds; SHA-256 verified against the
canonical Kaggle file). Nothing is removed or relabelled: 1,081 exact
duplicate rows are retained and the number of evaluation rows whose exact
feature vector also occurs in training data is recorded in the split
manifest (299 test rows on the full data). The banks are **simulated
partitions** of this single dataset; ULB has no institution identifiers.

Splits (one fixed split seed = 42, identical for every run seed, scenario and
condition):

1. 80/20 stratified development/test.
2. 15 % of development → validation (stratified).
3. 10 % of the remainder → public proxy/reference set. **Primary:
   stratified** — this is *label-informed* construction of the reference
   set (labels are used to stratify, then dropped; they are never written to
   the federation view). **Sensitivity:** `configs/paper_random_proxy.yaml`
   draws the proxy uniformly (natural class composition).
4. The rest is the client pool, partitioned among Banks A/B/C.

Run seeds (42, 123, 2026) change the client partition and model randomness,
not the test set. Seed-to-seed spread is therefore training variability
only; test-sample uncertainty is estimated separately by bootstrap.

**Preprocessing** (all conditions, all models): `Amount → log1p(Amount)`,
then per-feature standardisation with statistics from the **proxy features
only** (a permitted public source, so no private client data is pooled and
validation/test are never used). Trees are invariant to it; FedAvg needs the
identical transform on every client, which this guarantees.

## Heterogeneity scenarios

* **IID**: per-class shuffling and equal splitting.
* **Dirichlet(α)**, class-conditioned: for each class *c*, proportions
  *q_c ~ Dir(α·1₃)* split that class's rows among the banks. α = 1.0 (mild,
  ablation E), 0.5 (moderate), 0.1 (severe).
* Each bank must receive ≥ `min_fraud_per_client` frauds and
  ≥ `min_legit_per_client` legitimate rows (paper: 10 / 1,000). Violating
  draws are rejected and redrawn deterministically (≤ 1,000 attempts); if no
  draw qualifies the run fails with a diagnostic, it is never relaxed. The
  accepted partition is a draw from the Dirichlet scheme *conditioned on the
  constraints*.
* α does not fix the degree of heterogeneity: with only three clients a
  single draw can be more or less extreme than α suggests, and the class-wise
  Dirichlet also produces strong **quantity** skew (e.g. one bank with < 2 %
  of the data). Analyses must use the realised fraud rates and client shares
  (Table 1: fraud-rate CV and largest client share per draw).

## Models (fixed a priori, `configs/models.yaml`; not tuned)

| Bank | Model | Key settings |
|---|---|---|
| A | XGBoost 3.4.1 | 300 trees, depth 6, η 0.1, subsample/colsample 0.8, `hist`, `binary:logistic` |
| B | scikit-learn RandomForest | 200 trees, `max_features=sqrt`, bootstrap |
| C (and every FedAvg client) | PyTorch MLP | 30→64→32→1, ReLU, dropout 0.1, Adam 1e-3, batch 512, 10 epochs |

No class re-weighting is applied to any model, so all predicted probabilities
live on the natural likelihood scale that the KD teachers average;
operating points are handled by the validation-selected threshold.

## Conditions

1. **Local-only** — each bank trains its model on its own partition.
2. **Centralized (reference, not federated)** — each model family trained on
   the union of the three partitions (no proxy/validation/test rows). Not
   assumed to be an upper bound.
3. **FedAvg** (Flower) — all banks train the same MLP; server aggregates
   `θ ← Σ_k (n_k / Σ_j n_j) θ_k` over 5 rounds × 2 local epochs (= the
   10-epoch local budget). Fresh Adam state each round. Global model
   validated after every round on the server.
4. **Heterogeneous ensemble, no KD** — the local-only models of condition 1,
   combined at prediction time by an equal-weight probability average.
5. **Heterogeneous KD (proposed)** — below.

## Knowledge-distillation protocol (FedMD-inspired, per-example proxy distillation)

Round 0: each bank trains its model on private labelled data only (identical
to condition 1 — verified: max |Δp| = 0 on validation), then sends its fraud
probabilities *p_k(x)* for the ordered proxy set *P* (|P| = 19,367).

Server, each round r ≥ 1 (equal weights, leave-one-out):

```
T_A(x) = (p_B(x) + p_C(x)) / 2,   T_B(x) = (p_A(x) + p_C(x)) / 2,   T_C(x) = (p_A(x) + p_B(x)) / 2
```

Each bank receives only its own teacher. No parameters are aggregated and no
global student exists. Ablations: full-committee teacher (includes the
student's own prediction) and validation-AP-weighted teachers.

Client update with `λ = lambda_kd` (primary 0.5), private labels remaining
the supervised signal:

* **MLP** (incremental): fine-tune the current weights for 2 epochs on
  `L = mean_batch BCE(f(x_i), y_i) + λ · mean_proxybatch BCE(f(x_j), clip(T(x_j)))`
  with `BCEWithLogitsLoss` semantics and `clip = [1e-6, 1−1e-6]`. One proxy
  mini-batch (512) per local step. Both terms are per-example means, so the
  proxy cannot dominate by size.
* **XGBoost** (retrained from scratch, same seed, 300 trees): local rows with
  weight 1 plus, for every proxy example, `(x, 1, w·p)` and `(x, 0, w·(1−p))`,
  with `w = λ·N_local/|P|`. Total proxy weight = λ·N_local. Because XGBoost
  minimises a sample-weighted log-loss, this is *exactly*
  `Σ_local BCE + Σ_proxy w·softBCE(p, q)`.
* **Random Forest** (retrained from scratch — no incremental fit exists):
  the same weighted expansion, fitted with `sample_weight`. In scikit-learn
  1.9.1 the bootstrap then samples rows ∝ weight; `max_samples=1.0` makes each
  tree draw `Σw = N_local(1+λ)` rows, i.e. in expectation the same local data
  as the local-only forest plus λ·N_local proxy draws whose labels follow the
  teacher. This *approximates* soft-target training (each tree sees hard
  labels sampled from T; the forest averages them). With λ = 0 no weights are
  passed and the local-only forest is reproduced exactly.

Hard pseudo-labels are not used (with 0.17 % fraud they would make almost all
proxy rows negative). After each round, each bank's validation predictions
are stored; the committee of distilled models is combined exactly like
condition 4.

## Evaluation

* **Primary metric:** Average Precision (`sklearn.metrics.average_precision_score`),
  reported as "AP". The trapezoidal PR-AUC is reported separately as
  "PR-AUC (trap.)".
* Secondary: ROC-AUC; precision, recall, F1, balanced accuracy, FPR,
  FP/FN counts and confusion matrix at threshold 0.5 **and** at the
  validation-selected threshold; recall at FPR ≤ 0.1 % and ≤ 1 %; training
  and inference time; communication bytes.
* **Threshold rule** (fixed before any test evaluation, same for every
  system): the threshold maximising F1 on the validation split.
* **Model selection** (fixed a priori): the final round (5) for FedAvg and
  KD. `best_validation_ap` is available but not used for the primary results.
  Test data are touched only after selection, by the evaluation harness.

## Statistics

* Per (scenario, system): mean, sample SD and a t-interval across the 3 seeds
  — a description of training variability, **not** independent replications
  (seeds share the test set).
* Per (seed, scenario): class-stratified bootstrap (positives and negatives
  resampled separately, 1,000 replicates) CIs for AP, and paired bootstrap CIs
  for AP differences between systems on the same resamples.
* Seed-paired differences are summarised descriptively (mean, range, number
  of seeds favouring each system). No hypothesis tests are reported; with
  three seeds on a 98-fraud test set, claims of significance would not be
  justified.
