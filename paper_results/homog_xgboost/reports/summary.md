# Experiment report: `homog_xgboost`

## Provenance

- Dataset SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` (matches canonical ULB release: True)
- Experiment identity: `4dc5f04760b4d0a65e6a12fc229ef23826b2717c3ae20b81baa7c0a49cc316b8`; configuration hash: `9e8985b1bc0d8cd40aa25a1f409b132fcbcd732c1d48fc5781f08c8c5a224654`
- Source revision(s) of the completed runs: `3a0f650006c3`
- Source provenance verified for every run: True
- Python 3.11.0, flwr 1.26.1, xgboost 3.2.0, torch 2.14.1, scikit-learn 1.9.1
- Runs completed: 48; not completed: 0

## Test AP by system (mean over seeds; SDs in `tables/table7_seed_summary.csv`)

| short_label | IID | Mild (α=1.0) | Moderate (α=0.5) | Severe (α=0.1) |
|---|---|---|---|---|
| Local A (XGB) | 0.8573 | 0.8323 | 0.8357 | 0.8260 |
| Local B (XGB) | 0.8661 | 0.8547 | 0.8555 | 0.7864 |
| Local C (XGB) | 0.8729 | 0.8778 | 0.8653 | 0.8721 |
| Ensemble (equal) | 0.8801 | 0.8742 | 0.8769 | 0.8784 |
| Fed. XGB bagging | 0.8410 | 0.8419 | 0.8561 | 0.8469 |
| KD A (XGB) | 0.8765 | 0.8588 | 0.8411 | 0.8515 |
| KD B (XGB) | 0.8712 | 0.8646 | 0.8713 | 0.8172 |
| KD C (XGB) | 0.8828 | 0.8787 | 0.8797 | 0.8770 |
| KD committee | 0.8870 | 0.8803 | 0.8813 | 0.8812 |

Seeds share one test set, so the spread across seeds reflects training variability (partition + initialisation) only, not test-sample uncertainty. Test-sample uncertainty is in `reports/bootstrap_*.csv` (class-stratified bootstrap).

## Seed-paired AP differences (a − b), descriptive

| scenario | a | b | n_pairs | mean_diff | std_diff | min_diff | max_diff | n_a_better |
|---|---|---|---|---|---|---|---|---|
| iid | kd_committee | ensemble_equal | 3 | 0.0069 | 0.0034 | 0.0045 | 0.0108 | 3 |
| mild_non_iid | kd_committee | ensemble_equal | 3 | 0.0061 | 0.0073 | -0.0003 | 0.0140 | 2 |
| moderate_non_iid | kd_committee | ensemble_equal | 3 | 0.0044 | 0.0024 | 0.0017 | 0.0062 | 3 |
| severe_non_iid | kd_committee | ensemble_equal | 3 | 0.0029 | 0.0014 | 0.0014 | 0.0042 | 3 |
| iid | kd_bank_A | bank_A | 3 | 0.0192 | 0.0078 | 0.0125 | 0.0277 | 3 |
| mild_non_iid | kd_bank_A | bank_A | 3 | 0.0265 | 0.0479 | -0.0066 | 0.0814 | 2 |
| moderate_non_iid | kd_bank_A | bank_A | 3 | 0.0053 | 0.0109 | -0.0061 | 0.0154 | 2 |
| severe_non_iid | kd_bank_A | bank_A | 3 | 0.0254 | 0.0256 | -0.0029 | 0.0468 | 2 |
| iid | kd_bank_B | bank_B | 3 | 0.0051 | 0.0157 | -0.0129 | 0.0159 | 2 |
| mild_non_iid | kd_bank_B | bank_B | 3 | 0.0098 | 0.0125 | -0.0045 | 0.0183 | 2 |
| moderate_non_iid | kd_bank_B | bank_B | 3 | 0.0158 | 0.0043 | 0.0124 | 0.0206 | 3 |
| severe_non_iid | kd_bank_B | bank_B | 3 | 0.0308 | 0.0285 | 0.0016 | 0.0584 | 3 |
| iid | kd_bank_C | bank_C | 3 | 0.0099 | 0.0079 | 0.0021 | 0.0178 | 3 |
| mild_non_iid | kd_bank_C | bank_C | 3 | 0.0009 | 0.0097 | -0.0047 | 0.0121 | 1 |
| moderate_non_iid | kd_bank_C | bank_C | 3 | 0.0145 | 0.0099 | 0.0085 | 0.0259 | 3 |
| severe_non_iid | kd_bank_C | bank_C | 3 | 0.0049 | 0.0144 | -0.0058 | 0.0213 | 1 |
| iid | kd_committee | fedxgb_global | 3 | 0.0460 | 0.0077 | 0.0372 | 0.0518 | 3 |
| mild_non_iid | kd_committee | fedxgb_global | 3 | 0.0383 | 0.0081 | 0.0307 | 0.0469 | 3 |
| moderate_non_iid | kd_committee | fedxgb_global | 3 | 0.0251 | 0.0168 | 0.0101 | 0.0433 | 3 |
| severe_non_iid | kd_committee | fedxgb_global | 3 | 0.0344 | 0.0155 | 0.0168 | 0.0462 | 3 |
| iid | ensemble_equal | fedxgb_global | 3 | 0.0391 | 0.0078 | 0.0318 | 0.0473 | 3 |
| mild_non_iid | ensemble_equal | fedxgb_global | 3 | 0.0322 | 0.0152 | 0.0167 | 0.0471 | 3 |
| moderate_non_iid | ensemble_equal | fedxgb_global | 3 | 0.0207 | 0.0154 | 0.0084 | 0.0380 | 3 |
| severe_non_iid | ensemble_equal | fedxgb_global | 3 | 0.0315 | 0.0144 | 0.0154 | 0.0432 | 3 |

Statements generated from the table above (no significance testing):

- IID: `kd_committee` AP is on average 0.0069 higher than `ensemble_equal` (range +0.0045 to +0.0108; `kd_committee` higher in 3/3 seeds).
- Mild (α=1.0): `kd_committee` AP is on average 0.0061 higher than `ensemble_equal` (range -0.0003 to +0.0140; `kd_committee` higher in 2/3 seeds).
- Moderate (α=0.5): `kd_committee` AP is on average 0.0044 higher than `ensemble_equal` (range +0.0017 to +0.0062; `kd_committee` higher in 3/3 seeds).
- Severe (α=0.1): `kd_committee` AP is on average 0.0029 higher than `ensemble_equal` (range +0.0014 to +0.0042; `kd_committee` higher in 3/3 seeds).
- IID: `kd_bank_A` AP is on average 0.0192 higher than `bank_A` (range +0.0125 to +0.0277; `kd_bank_A` higher in 3/3 seeds).
- Mild (α=1.0): `kd_bank_A` AP is on average 0.0265 higher than `bank_A` (range -0.0066 to +0.0814; `kd_bank_A` higher in 2/3 seeds).
- Moderate (α=0.5): `kd_bank_A` AP is on average 0.0053 higher than `bank_A` (range -0.0061 to +0.0154; `kd_bank_A` higher in 2/3 seeds).
- Severe (α=0.1): `kd_bank_A` AP is on average 0.0254 higher than `bank_A` (range -0.0029 to +0.0468; `kd_bank_A` higher in 2/3 seeds).
- IID: `kd_bank_B` AP is on average 0.0051 higher than `bank_B` (range -0.0129 to +0.0159; `kd_bank_B` higher in 2/3 seeds).
- Mild (α=1.0): `kd_bank_B` AP is on average 0.0098 higher than `bank_B` (range -0.0045 to +0.0183; `kd_bank_B` higher in 2/3 seeds).
- Moderate (α=0.5): `kd_bank_B` AP is on average 0.0158 higher than `bank_B` (range +0.0124 to +0.0206; `kd_bank_B` higher in 3/3 seeds).
- Severe (α=0.1): `kd_bank_B` AP is on average 0.0308 higher than `bank_B` (range +0.0016 to +0.0584; `kd_bank_B` higher in 3/3 seeds).
- IID: `kd_bank_C` AP is on average 0.0099 higher than `bank_C` (range +0.0021 to +0.0178; `kd_bank_C` higher in 3/3 seeds).
- Mild (α=1.0): `kd_bank_C` AP is on average 0.0009 higher than `bank_C` (range -0.0047 to +0.0121; `kd_bank_C` higher in 1/3 seeds).
- Moderate (α=0.5): `kd_bank_C` AP is on average 0.0145 higher than `bank_C` (range +0.0085 to +0.0259; `kd_bank_C` higher in 3/3 seeds).
- Severe (α=0.1): `kd_bank_C` AP is on average 0.0049 higher than `bank_C` (range -0.0058 to +0.0213; `kd_bank_C` higher in 1/3 seeds).
- IID: `kd_committee` AP is on average 0.0460 higher than `fedxgb_global` (range +0.0372 to +0.0518; `kd_committee` higher in 3/3 seeds).
- Mild (α=1.0): `kd_committee` AP is on average 0.0383 higher than `fedxgb_global` (range +0.0307 to +0.0469; `kd_committee` higher in 3/3 seeds).
- Moderate (α=0.5): `kd_committee` AP is on average 0.0251 higher than `fedxgb_global` (range +0.0101 to +0.0433; `kd_committee` higher in 3/3 seeds).
- Severe (α=0.1): `kd_committee` AP is on average 0.0344 higher than `fedxgb_global` (range +0.0168 to +0.0462; `kd_committee` higher in 3/3 seeds).
- IID: `ensemble_equal` AP is on average 0.0391 higher than `fedxgb_global` (range +0.0318 to +0.0473; `ensemble_equal` higher in 3/3 seeds).
- Mild (α=1.0): `ensemble_equal` AP is on average 0.0322 higher than `fedxgb_global` (range +0.0167 to +0.0471; `ensemble_equal` higher in 3/3 seeds).
- Moderate (α=0.5): `ensemble_equal` AP is on average 0.0207 higher than `fedxgb_global` (range +0.0084 to +0.0380; `ensemble_equal` higher in 3/3 seeds).
- Severe (α=0.1): `ensemble_equal` AP is on average 0.0315 higher than `fedxgb_global` (range +0.0154 to +0.0432; `ensemble_equal` higher in 3/3 seeds).

## Paired bootstrap of test AP differences (per seed)

| seed | scenario | a | b | diff | ci_low | ci_high | ci_excludes_zero |
|---|---|---|---|---|---|---|---|
| 42 | iid | kd_committee | ensemble_equal | 0.0055 | -0.0029 | 0.0143 | False |
| 42 | iid | kd_bank_A | bank_A | 0.0174 | 0.0027 | 0.0431 | True |
| 42 | iid | kd_bank_B | bank_B | -0.0129 | -0.0402 | 0.0091 | False |
| 42 | iid | kd_bank_C | bank_C | 0.0178 | 0.0054 | 0.0335 | True |
| 42 | iid | kd_committee | fedxgb_global | 0.0372 | 0.0147 | 0.0644 | True |
| 42 | iid | ensemble_equal | fedxgb_global | 0.0318 | 0.0079 | 0.0603 | True |
| 42 | mild_non_iid | kd_committee | ensemble_equal | -0.0003 | -0.0067 | 0.0074 | False |
| 42 | mild_non_iid | kd_bank_A | bank_A | -0.0066 | -0.0160 | 0.0017 | False |
| 42 | mild_non_iid | kd_bank_B | bank_B | 0.0183 | -0.0081 | 0.0429 | False |
| 42 | mild_non_iid | kd_bank_C | bank_C | -0.0047 | -0.0120 | 0.0009 | False |
| 42 | mild_non_iid | kd_committee | fedxgb_global | 0.0469 | 0.0154 | 0.0883 | True |
| 42 | mild_non_iid | ensemble_equal | fedxgb_global | 0.0471 | 0.0154 | 0.0892 | True |
| 42 | moderate_non_iid | kd_committee | ensemble_equal | 0.0017 | -0.0051 | 0.0079 | False |
| 42 | moderate_non_iid | kd_bank_A | bank_A | 0.0154 | -0.0217 | 0.0405 | False |
| 42 | moderate_non_iid | kd_bank_B | bank_B | 0.0206 | 0.0067 | 0.0401 | True |
| 42 | moderate_non_iid | kd_bank_C | bank_C | 0.0090 | -0.0115 | 0.0419 | False |
| 42 | moderate_non_iid | kd_committee | fedxgb_global | 0.0101 | -0.0076 | 0.0289 | False |
| 42 | moderate_non_iid | ensemble_equal | fedxgb_global | 0.0084 | -0.0089 | 0.0273 | False |
| 42 | severe_non_iid | kd_committee | ensemble_equal | 0.0014 | -0.0069 | 0.0108 | False |
| 42 | severe_non_iid | kd_bank_A | bank_A | 0.0323 | 0.0080 | 0.0600 | True |
| 42 | severe_non_iid | kd_bank_B | bank_B | 0.0584 | 0.0212 | 0.1028 | True |
| 42 | severe_non_iid | kd_bank_C | bank_C | -0.0058 | -0.0168 | 0.0042 | False |
| 42 | severe_non_iid | kd_committee | fedxgb_global | 0.0168 | -0.0090 | 0.0475 | False |
| 42 | severe_non_iid | ensemble_equal | fedxgb_global | 0.0154 | -0.0133 | 0.0504 | False |
| 123 | iid | kd_committee | ensemble_equal | 0.0108 | 0.0028 | 0.0206 | True |
| 123 | iid | kd_bank_A | bank_A | 0.0277 | 0.0095 | 0.0523 | True |
| 123 | iid | kd_bank_B | bank_B | 0.0124 | -0.0035 | 0.0309 | False |
| 123 | iid | kd_bank_C | bank_C | 0.0099 | 0.0017 | 0.0192 | True |
| 123 | iid | kd_committee | fedxgb_global | 0.0490 | 0.0183 | 0.0927 | True |
| 123 | iid | ensemble_equal | fedxgb_global | 0.0382 | 0.0066 | 0.0817 | True |
| 123 | mild_non_iid | kd_committee | ensemble_equal | 0.0140 | 0.0005 | 0.0332 | True |
| 123 | mild_non_iid | kd_bank_A | bank_A | 0.0814 | 0.0446 | 0.1211 | True |
| 123 | mild_non_iid | kd_bank_B | bank_B | -0.0045 | -0.0244 | 0.0163 | False |
| 123 | mild_non_iid | kd_bank_C | bank_C | 0.0121 | -0.0032 | 0.0299 | False |
| 123 | mild_non_iid | kd_committee | fedxgb_global | 0.0307 | -0.0052 | 0.0731 | False |
| 123 | mild_non_iid | ensemble_equal | fedxgb_global | 0.0167 | -0.0218 | 0.0606 | False |
| 123 | moderate_non_iid | kd_committee | ensemble_equal | 0.0053 | 0.0008 | 0.0111 | True |
| 123 | moderate_non_iid | kd_bank_A | bank_A | 0.0067 | -0.0002 | 0.0157 | False |
| 123 | moderate_non_iid | kd_bank_B | bank_B | 0.0124 | -0.0027 | 0.0292 | False |
| 123 | moderate_non_iid | kd_bank_C | bank_C | 0.0085 | -0.0030 | 0.0241 | False |
| 123 | moderate_non_iid | kd_committee | fedxgb_global | 0.0433 | 0.0174 | 0.0710 | True |
| 123 | moderate_non_iid | ensemble_equal | fedxgb_global | 0.0380 | 0.0131 | 0.0644 | True |
| 123 | severe_non_iid | kd_committee | ensemble_equal | 0.0042 | -0.0027 | 0.0122 | False |
| 123 | severe_non_iid | kd_bank_A | bank_A | -0.0029 | -0.0101 | 0.0051 | False |
| 123 | severe_non_iid | kd_bank_B | bank_B | 0.0325 | 0.0158 | 0.0521 | True |
| 123 | severe_non_iid | kd_bank_C | bank_C | 0.0213 | 0.0001 | 0.0488 | True |
| 123 | severe_non_iid | kd_committee | fedxgb_global | 0.0401 | 0.0154 | 0.0704 | True |
| 123 | severe_non_iid | ensemble_equal | fedxgb_global | 0.0359 | 0.0129 | 0.0640 | True |
| 2026 | iid | kd_committee | ensemble_equal | 0.0045 | -0.0012 | 0.0125 | False |
| 2026 | iid | kd_bank_A | bank_A | 0.0125 | -0.0033 | 0.0319 | False |
| 2026 | iid | kd_bank_B | bank_B | 0.0159 | 0.0037 | 0.0378 | True |
| 2026 | iid | kd_bank_C | bank_C | 0.0021 | -0.0041 | 0.0096 | False |
| 2026 | iid | kd_committee | fedxgb_global | 0.0518 | 0.0156 | 0.1029 | True |
| 2026 | iid | ensemble_equal | fedxgb_global | 0.0473 | 0.0094 | 0.0982 | True |
| 2026 | mild_non_iid | kd_committee | ensemble_equal | 0.0045 | 0.0003 | 0.0104 | True |
| 2026 | mild_non_iid | kd_bank_A | bank_A | 0.0047 | -0.0034 | 0.0148 | False |
| 2026 | mild_non_iid | kd_bank_B | bank_B | 0.0157 | 0.0022 | 0.0313 | True |
| 2026 | mild_non_iid | kd_bank_C | bank_C | -0.0046 | -0.0130 | 0.0023 | False |
| 2026 | mild_non_iid | kd_committee | fedxgb_global | 0.0374 | 0.0152 | 0.0667 | True |
| 2026 | mild_non_iid | ensemble_equal | fedxgb_global | 0.0329 | 0.0130 | 0.0587 | True |
| 2026 | moderate_non_iid | kd_committee | ensemble_equal | 0.0062 | -0.0016 | 0.0163 | False |
| 2026 | moderate_non_iid | kd_bank_A | bank_A | -0.0061 | -0.0135 | 0.0002 | False |
| 2026 | moderate_non_iid | kd_bank_B | bank_B | 0.0144 | 0.0033 | 0.0302 | True |
| 2026 | moderate_non_iid | kd_bank_C | bank_C | 0.0259 | 0.0031 | 0.0563 | True |
| 2026 | moderate_non_iid | kd_committee | fedxgb_global | 0.0220 | 0.0062 | 0.0417 | True |
| 2026 | moderate_non_iid | ensemble_equal | fedxgb_global | 0.0158 | 0.0008 | 0.0325 | True |
| 2026 | severe_non_iid | kd_committee | ensemble_equal | 0.0030 | -0.0088 | 0.0156 | False |
| 2026 | severe_non_iid | kd_bank_A | bank_A | 0.0468 | 0.0189 | 0.0776 | True |
| 2026 | severe_non_iid | kd_bank_B | bank_B | 0.0016 | -0.0158 | 0.0161 | False |
| 2026 | severe_non_iid | kd_bank_C | bank_C | -0.0009 | -0.0109 | 0.0104 | False |
| 2026 | severe_non_iid | kd_committee | fedxgb_global | 0.0462 | 0.0075 | 0.0949 | True |
| 2026 | severe_non_iid | ensemble_equal | fedxgb_global | 0.0432 | -0.0002 | 0.0999 | False |

Each interval reflects test-example resampling for one trained pair of models; intervals from different seeds are not independent replications. `ci_excludes_zero` is descriptive: it is not a significance test, no multiple-comparison correction is applied, and it says nothing about variation across training seeds or partitions.

## Ablations (test AP; committee and mean bank model)

| variant | scenario | committee_mean | committee_std | bank_mean | bank_std | n |
|---|---|---|---|---|---|---|
| ensemble:primary | iid | 0.8801 | 0.0031 | nan | nan | 3 |
| ensemble:primary | mild_non_iid | 0.8742 | 0.0165 | nan | nan | 3 |
| ensemble:primary | moderate_non_iid | 0.8769 | 0.0007 | nan | nan | 3 |
| ensemble:primary | severe_non_iid | 0.8784 | 0.0020 | nan | nan | 3 |
| primary | iid | 0.8870 | 0.0009 | 0.8769 | 0.0029 | 3 |
| primary | mild_non_iid | 0.8803 | 0.0097 | 0.8674 | 0.0098 | 3 |
| primary | moderate_non_iid | 0.8813 | 0.0019 | 0.8640 | 0.0209 | 3 |
| primary | severe_non_iid | 0.8812 | 0.0006 | 0.8486 | 0.0260 | 3 |

## Figures

- fig1: fig1_fraud_distribution.png, fig1_fraud_distribution.pdf
- fig2: fig2_ap_comparison.png, fig2_ap_comparison.pdf
- fig3: fig3_heterogeneity.png, fig3_heterogeneity.pdf
- fig4: fig4_validation_rounds.png, fig4_validation_rounds.pdf
- fig5: fig5_pr_curves.png, fig5_pr_curves.pdf
- fig6: fig6_roc_curves.png, fig6_roc_curves.pdf
- fig7: fig7_before_after_kd.png, fig7_before_after_kd.pdf
- fig8: fig8_cost.png, fig8_cost.pdf

## Tables

- `table1a_splits.tex`
- `table1b_partitions.tex`
- `table2_local.tex`
- `table3_main.tex`
- `table4_non_iid.tex`
- `table5_per_bank.tex`
- `table6_cost.tex`
- `table7_seed_summary.tex`
- `table8_ablations.tex`
