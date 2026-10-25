# Validation report: Heterogeneous Federated Committees for credit-card fraud

Status: all experiments below ran from one clean commit (49a2bc6, `source_dirty=false`) on the ULB dataset
(SHA-256 verified). Primary metric: test average precision (AP). Seeds in the paper profile: 42/123/2026 (partition
and model randomness); "split" = an independent draw of test/validation/proxy sets. All numbers are in
`paper_results/validation/`. Tests: 218 fast + 11 slow pass; the 1,081 duplicates and PCA caveats are unchanged.

## Bottom line
1. **The paper's pipeline reproduces.** Linux re-run of `configs/paper.yaml` matches the committed Windows numbers
   closely (e.g. KD committee minus lambda=0: +0.003/+0.005/-0.015 vs +0.001/+0.004/-0.016 for IID/moderate/severe),
   and the independent data audit found no leakage (disjoint splits, scaler fitted on proxy features only, no test rows
   or labels in any of the 12 federation views).
2. **"Committee beats FedAvg" is true but mostly a baseline artefact.** Over 11 independent splits (99 cells) the KD
   committee beats FedAvg-MLP by +0.052 AP (Holm p<0.001) -- but so does the *plain ensemble of local models*
   (+0.046), and the gap closes with a fairer baseline: FedAvg given 10 rounds -> +0.017 (1 round +0.152, 3 rounds
   +0.055, 5 rounds +0.038); tuned MLP (validation-selected) -> +0.014 with ensemble == KD committee (-0.0004).
   The default FedAvg MLP is also the weakest model family (centralised MLP 0.79 vs centralised XGBoost 0.855 in IID).
3. **Knowledge distillation adds nothing measurable at the committee level.** Against the matched control (lambda=0:
   same retraining/extra epochs, no teacher): KD committee - lambda0 committee = **-0.0048** pooled (cluster CI
   [-0.0076, -0.0020], mixed-model Holm p=0.049), driven by severe non-IID (-0.016); IID +0.0003, moderate +0.0013
   (both n.s.). The committee's gain over the no-KD ensemble (+0.006 pooled, Holm p=0.004) is explained by the lambda0
   committee already beating the ensemble by +0.011: it comes from the MLP's extra training epochs, not from distillation.
4. **Per-bank effects are real but narrow and not a model-family story.** Bank-level KD vs lambda0 (11 splits): XGBoost
   +0.011 (Holm p=0.018; +0.024 under severe skew), Random Forest +0.004 (n.s.), MLP +0.006 (n.s.; +0.017 IID, -0.012
   severe). Across all 6 bank<->model assignments (162 bank-runs), 70% of the MLP's apparent KD gain over local
   (+0.059) is the extra-epochs effect (lambda0 +0.041; true KD +0.018), and the KD gain shrinks with fraud count
   (coef -0.037 per log10 fraud, p<0.001): KD helps *fraud-poor* banks, whatever model they run.
5. **Random-split numbers are optimistic and the ranking shifts.** Temporal hold-out (train<val<test in Time): committee
   AP 0.88 -> 0.81, centralised XGBoost 0.887 -> 0.78 (the federated committee then beats the centralised XGBoost). Masking
   `Time` changes nothing (|dAP|<=0.004), so the models do not exploit time directly. Removing exact duplicates lowers AP
   by ~0.03 (committee) and destabilises the MLP bank (changes both train and test sets, so not strictly paired). Frauds
   with an exact training twin (5 of 98 test frauds) are detected perfectly, a small inflation.
6. **Sensitivity.** Proxy size (1.7k-48k rows) and rounds (1-10) never produce a KD-over-lambda0 committee gain
   (|diff|<=0.005). A fraud-poor random 1% proxy makes the XGBoost bank slightly worse (-0.010). Tuning: the MLP default is
   badly under-tuned (validation AP 0.798 -> 0.883); Random Forest defaults were not beaten; a tuned XGBoost (lr 0.2) tuned on
   IID-sized banks collapses on tiny partitions (bank A severe 0.83 -> 0.55): hyperparameters do not transfer across data sizes.
7. **Other.** KD slightly improves calibration of the RF/MLP members (ECE 0.0014 -> 0.0010 RF); a validation-fitted
   stacked logistic combiner is +0.005-0.007 AP over the plain mean (exploratory); max/rank combiners are worse. At a
   FN:FP cost of 500:1 the committee costs ~2-7% less than the best single bank chosen on validation. Per-seed rankings
   are unstable (mean Kendall tau 0.32 across seeds).

## Recommended wording changes for the paper
* Replace "heterogeneous KD committee outperforms homogeneous federated learning" with: a committee of diverse models
  outperforms a FedAvg MLP baseline, and so does simple prediction averaging; KD itself shows no committee-level benefit
  over a matched no-teacher control, but gives modest gains to individual fraud-poor banks (XGBoost most clearly).
* Report FedAvg with a tuned MLP and more rounds, state the baseline is under-converged at 5 rounds, include a
  homogeneous XGBoost baseline in the main table, and report lambda=0 as the control for every KD claim.
* Use multiple outer splits for uncertainty (single-split seed spread understates test variance) and report a temporal
  hold-out as the headline realistic estimate.
* Fix: broken citation "(?)"; Chen 2025a/2025b and Lei 2023a/2023b are duplicate entries; give primary sources for loss
  statistics; the methods text should describe the implemented leave-one-out FedMD-style protocol, rounds, lambda and
  hyperparameters; add the results section (the PDF has none).

## Caveats
* Statistics: 11 splits x 9 cells; cluster = split; exact sign-flip tests, cluster bootstrap CIs, Holm per scope; mixed
  model `diff ~ 1 + (1|split)`. Single-split groups (tuned, permutations, stress tests, sweeps) are descriptive over 9 cells.
* Temporal split has one deterministic split (~75 test frauds); tuned configuration was selected on split-42 validation.
* The dedup run changes both train and test sets; do not read its deltas as paired.
* Simulated banks from one public dataset, 30 PCA features, no privacy guarantee, no adversaries; unchanged from
  `docs/limitations.md`.

## Where to look
step1_tuned, step2_splits (campaign_analysis.md, split_level_comparisons.csv, mixed_model.csv), step3_perms,
step4_stress, step5_sweeps / step5_rounds, step6_posthoc/step6.md, tuning/tuning_trials.csv, reanalysis_committed and
data_audit (from the committed-results re-analysis and independent data audit). Performance: `docs/performance_notes.md`.
