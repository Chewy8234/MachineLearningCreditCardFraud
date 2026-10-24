# Limitations and interpretation

## Data and simulation

* **Simulated institutions.** Banks A/B/C are partitions of one public dataset
  (two days of European card transactions, 2013). ULB has no institution
  identifiers; the experiment does not demonstrate real cross-institution
  deployment, and results do not automatically generalise to production
  fraud systems, other time periods or other feature spaces.
* **Shared feature space.** All banks use the same 30 PCA-derived features
  (V1–V28 are already anonymised principal components computed on the whole
  dataset by its publishers). Real institutions would not share such a
  representation.
* **Proxy set assumption.** The public reference set is carved from ULB.
  The primary proxy is *label-informed* (stratified on the label before
  labels are dropped), so it has the population fraud rate (34 frauds in
  19,367 rows). A naturally sampled proxy is provided as a sensitivity
  analysis, but at this size a uniform sample lands at almost the same fraud
  count (35), so it tests the *draw* of the proxy set rather than a different
  composition; a deliberately fraud-poor or fraud-rich proxy was not tested.
  The per-bank KD effects changed noticeably (including sign, for the
  Random Forest under severe skew) between the two proxy draws. A real public
  reference set may differ from the clients' distribution.
* **Simulation setup vs protocol.** The experiment-preparation code reads
  the whole dataset to build partitions; the federation itself only reads
  each bank's file plus the label-free proxy and the validation file. In
  the simulation, nothing *prevents* a client process from opening another
  bank's file — isolation is by construction of the code, not by OS
  sandboxing.
* **Validation set location.** Validation data are held by the experimenter
  and used by clients only to *report* per-round metrics (and, in ablation D,
  to weight teachers). A real deployment would need its own validation data
  per party.
* **Duplicates.** 1,081 exact duplicate rows are retained (no silent
  removal); 299 test rows have an identical feature vector in the training
  data. This may slightly favour memorising models (e.g. fully grown random
  forests) in all conditions alike.
* **Few positives.** The test set has 98 frauds and the validation set 59.
  A difference of one or two detected frauds moves recall by 1–2 points and
  AP noticeably; bootstrap intervals are wide.

## Heterogeneity

* Dirichlet α does not determine realised heterogeneity. With three clients,
  draws also produce large *quantity* skew (some banks hold < 2 % of the
  data), confounding label skew with data scarcity. Constraint-based
  rejection sampling (≥ 10 frauds, ≥ 1,000 legitimate per bank) conditions
  the distribution of partitions. Use Table 1's realised statistics, not α,
  when interpreting RQ2.
* Partitions differ by seed, so "severe" for one seed is not the same
  severity as for another.

## Methods

* **Random Forest soft targets are an approximation.** With the weighted
  bootstrap each tree sees hard labels sampled from the teacher; small
  teacher probabilities are rarely sampled at all.
* **XGBoost and RF are retrained every round**, the MLP is fine-tuned
  incrementally. The KD MLP therefore receives 10 more epochs than the
  local-only MLP; the λ = 0 ablation is the control for this extra training.
  Retraining makes the tree models' per-round cost independent of history
  but means a tree model's "state" is only its latest proxy predictions.
* **Fixed hyperparameters.** No model was tuned (not on validation, not on
  test). Different settings, class weighting or calibration could change
  the ranking of conditions. FedAvg uses a fresh optimiser state each round
  and no server momentum.
* **Model selection = final round** for all federated methods (fixed a
  priori). Validation curves are reported; a different stopping rule could
  change results.
* **Committee combination** is a plain probability average; probabilities of
  differently trained models are not calibrated to each other.

## Privacy and security

* Sharing per-example predictions on a public proxy set is **not** a formal
  privacy guarantee. Predictions can leak information about the training
  data (membership inference, model inversion); no differential privacy,
  secure aggregation or cryptographic protection is applied.
* No adversarial or Byzantine clients are modelled.

## Statistics

* Three seeds share one test set: seed variation is training variability,
  not independent replication, and no significance tests are claimed.
* Bootstrap intervals quantify test-sample uncertainty for a fixed trained
  model only.

## Runtime and communication measurements

* Times are wall-clock on one laptop (Apple Silicon, 12 cores) with the three
  clients simulated concurrently by Ray; they are indicative, not
  production estimates. Some runs of seed 42 executed while the automated
  test suite was also running on the same machine, inflating their times.
* Communication counts serialised array payloads (ArrayRecord bytes) in
  both directions; protocol overhead (gRPC/HTTP framing, config and metric
  records) is not included. Prediction vectors are sent as float32.
* The MLP is pinned to one CPU thread (see architecture notes), which
  understates what a multi-threaded MLP could achieve in time.
