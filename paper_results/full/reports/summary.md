# Experiment report: `full`

## Provenance

- Dataset SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` (matches canonical ULB release: True)
- Experiment identity: `2bdb6ea04a861c3a411652560eb2796aefe861ab8ce270cb648a89c2db52a6cf`; configuration hash: `cf2167083568f0361276c7d8182008d6a08dc4824f2dbf1f14cfe92e80d12ebb`
- Source revision(s) of the completed runs: `3a0f650006c3`
- Source provenance verified for every run: True
- Python 3.11.0, flwr 1.26.1, xgboost 3.2.0, torch 2.14.1, scikit-learn 1.9.1
- Runs completed: 111; not completed: 0

## Test AP by system (mean over seeds; SDs in `tables/table7_seed_summary.csv`)

| short_label | IID | Mild (α=1.0) | Moderate (α=0.5) | Severe (α=0.1) |
|---|---|---|---|---|
| Local A (XGB) | 0.8573 | 0.8323 | 0.8357 | 0.8260 |
| Local B (RF) | 0.8556 | 0.8394 | 0.8524 | 0.7842 |
| Local C (MLP) | 0.7977 | 0.8011 | 0.8432 | 0.8047 |
| Central. XGB (ref.) | 0.8865 | nan | 0.8865 | 0.8865 |
| Central. RF (ref.) | 0.8775 | nan | 0.8775 | 0.8775 |
| Central. MLP (ref.) | 0.8475 | nan | 0.8475 | 0.8475 |
| Ensemble (equal) | 0.8783 | 0.8658 | 0.8785 | 0.8595 |
| FedAvg MLP | 0.8430 | 0.8464 | 0.8312 | 0.8352 |
| KD A (XGB) | 0.8667 | 0.8582 | 0.8358 | 0.8374 |
| KD B (RF) | 0.8568 | 0.8514 | 0.8587 | 0.7687 |
| KD C (MLP) | 0.8250 | 0.8242 | 0.8473 | 0.8512 |
| KD committee | 0.8810 | 0.8752 | 0.8836 | 0.8537 |

Seeds share one test set, so the spread across seeds reflects training variability (partition + initialisation) only, not test-sample uncertainty. Test-sample uncertainty is in `reports/bootstrap_*.csv` (class-stratified bootstrap).

## Seed-paired AP differences (a − b), descriptive

| scenario | a | b | n_pairs | mean_diff | std_diff | min_diff | max_diff | n_a_better |
|---|---|---|---|---|---|---|---|---|
| iid | kd_committee | ensemble_equal | 3 | 0.0028 | 0.0079 | -0.0049 | 0.0108 | 2 |
| mild_non_iid | kd_committee | ensemble_equal | 3 | 0.0094 | 0.0247 | -0.0061 | 0.0379 | 1 |
| moderate_non_iid | kd_committee | ensemble_equal | 3 | 0.0051 | 0.0026 | 0.0025 | 0.0077 | 3 |
| severe_non_iid | kd_committee | ensemble_equal | 3 | -0.0058 | 0.0195 | -0.0173 | 0.0167 | 1 |
| iid | kd_committee | fedavg_global | 3 | 0.0380 | 0.0357 | 0.0088 | 0.0778 | 3 |
| mild_non_iid | kd_committee | fedavg_global | 3 | 0.0288 | 0.0269 | 0.0123 | 0.0599 | 3 |
| moderate_non_iid | kd_committee | fedavg_global | 3 | 0.0524 | 0.0367 | 0.0228 | 0.0935 | 3 |
| severe_non_iid | kd_committee | fedavg_global | 3 | 0.0185 | 0.0625 | -0.0417 | 0.0831 | 2 |
| iid | ensemble_equal | fedavg_global | 3 | 0.0353 | 0.0281 | 0.0137 | 0.0670 | 3 |
| mild_non_iid | ensemble_equal | fedavg_global | 3 | 0.0193 | 0.0032 | 0.0158 | 0.0219 | 3 |
| moderate_non_iid | ensemble_equal | fedavg_global | 3 | 0.0472 | 0.0361 | 0.0203 | 0.0882 | 3 |
| severe_non_iid | ensemble_equal | fedavg_global | 3 | 0.0243 | 0.0458 | -0.0244 | 0.0664 | 2 |
| iid | kd_bank_A | bank_A | 3 | 0.0094 | 0.0152 | -0.0079 | 0.0205 | 2 |
| mild_non_iid | kd_bank_A | bank_A | 3 | 0.0259 | 0.0640 | -0.0157 | 0.0996 | 1 |
| moderate_non_iid | kd_bank_A | bank_A | 3 | 0.0001 | 0.0152 | -0.0174 | 0.0098 | 2 |
| severe_non_iid | kd_bank_A | bank_A | 3 | 0.0114 | 0.0230 | -0.0074 | 0.0370 | 2 |
| iid | kd_bank_B | bank_B | 3 | 0.0012 | 0.0095 | -0.0074 | 0.0114 | 1 |
| mild_non_iid | kd_bank_B | bank_B | 3 | 0.0121 | 0.0033 | 0.0088 | 0.0155 | 3 |
| moderate_non_iid | kd_bank_B | bank_B | 3 | 0.0063 | 0.0075 | -0.0008 | 0.0142 | 2 |
| severe_non_iid | kd_bank_B | bank_B | 3 | -0.0155 | 0.0128 | -0.0275 | -0.0021 | 0 |
| iid | kd_bank_C | bank_C | 3 | 0.0273 | 0.0223 | 0.0019 | 0.0439 | 3 |
| mild_non_iid | kd_bank_C | bank_C | 3 | 0.0231 | 0.0511 | -0.0322 | 0.0684 | 2 |
| moderate_non_iid | kd_bank_C | bank_C | 3 | 0.0042 | 0.0064 | 0.0000 | 0.0115 | 3 |
| severe_non_iid | kd_bank_C | bank_C | 3 | 0.0465 | 0.0303 | 0.0123 | 0.0699 | 3 |

Statements generated from the table above (no significance testing):

- IID: `kd_committee` AP is on average 0.0028 higher than `ensemble_equal` (range -0.0049 to +0.0108; `kd_committee` higher in 2/3 seeds).
- Mild (α=1.0): `kd_committee` AP is on average 0.0094 higher than `ensemble_equal` (range -0.0061 to +0.0379; `kd_committee` higher in 1/3 seeds).
- Moderate (α=0.5): `kd_committee` AP is on average 0.0051 higher than `ensemble_equal` (range +0.0025 to +0.0077; `kd_committee` higher in 3/3 seeds).
- Severe (α=0.1): `kd_committee` AP is on average 0.0058 lower than `ensemble_equal` (range -0.0173 to +0.0167; `kd_committee` higher in 1/3 seeds).
- IID: `kd_committee` AP is on average 0.0380 higher than `fedavg_global` (range +0.0088 to +0.0778; `kd_committee` higher in 3/3 seeds).
- Mild (α=1.0): `kd_committee` AP is on average 0.0288 higher than `fedavg_global` (range +0.0123 to +0.0599; `kd_committee` higher in 3/3 seeds).
- Moderate (α=0.5): `kd_committee` AP is on average 0.0524 higher than `fedavg_global` (range +0.0228 to +0.0935; `kd_committee` higher in 3/3 seeds).
- Severe (α=0.1): `kd_committee` AP is on average 0.0185 higher than `fedavg_global` (range -0.0417 to +0.0831; `kd_committee` higher in 2/3 seeds).
- IID: `ensemble_equal` AP is on average 0.0353 higher than `fedavg_global` (range +0.0137 to +0.0670; `ensemble_equal` higher in 3/3 seeds).
- Mild (α=1.0): `ensemble_equal` AP is on average 0.0193 higher than `fedavg_global` (range +0.0158 to +0.0219; `ensemble_equal` higher in 3/3 seeds).
- Moderate (α=0.5): `ensemble_equal` AP is on average 0.0472 higher than `fedavg_global` (range +0.0203 to +0.0882; `ensemble_equal` higher in 3/3 seeds).
- Severe (α=0.1): `ensemble_equal` AP is on average 0.0243 higher than `fedavg_global` (range -0.0244 to +0.0664; `ensemble_equal` higher in 2/3 seeds).
- IID: `kd_bank_A` AP is on average 0.0094 higher than `bank_A` (range -0.0079 to +0.0205; `kd_bank_A` higher in 2/3 seeds).
- Mild (α=1.0): `kd_bank_A` AP is on average 0.0259 higher than `bank_A` (range -0.0157 to +0.0996; `kd_bank_A` higher in 1/3 seeds).
- Moderate (α=0.5): `kd_bank_A` AP is on average 0.0001 higher than `bank_A` (range -0.0174 to +0.0098; `kd_bank_A` higher in 2/3 seeds).
- Severe (α=0.1): `kd_bank_A` AP is on average 0.0114 higher than `bank_A` (range -0.0074 to +0.0370; `kd_bank_A` higher in 2/3 seeds).
- IID: `kd_bank_B` AP is on average 0.0012 higher than `bank_B` (range -0.0074 to +0.0114; `kd_bank_B` higher in 1/3 seeds).
- Mild (α=1.0): `kd_bank_B` AP is on average 0.0121 higher than `bank_B` (range +0.0088 to +0.0155; `kd_bank_B` higher in 3/3 seeds).
- Moderate (α=0.5): `kd_bank_B` AP is on average 0.0063 higher than `bank_B` (range -0.0008 to +0.0142; `kd_bank_B` higher in 2/3 seeds).
- Severe (α=0.1): `kd_bank_B` AP is on average 0.0155 lower than `bank_B` (range -0.0275 to -0.0021; `kd_bank_B` higher in 0/3 seeds).
- IID: `kd_bank_C` AP is on average 0.0273 higher than `bank_C` (range +0.0019 to +0.0439; `kd_bank_C` higher in 3/3 seeds).
- Mild (α=1.0): `kd_bank_C` AP is on average 0.0231 higher than `bank_C` (range -0.0322 to +0.0684; `kd_bank_C` higher in 2/3 seeds).
- Moderate (α=0.5): `kd_bank_C` AP is on average 0.0042 higher than `bank_C` (range +0.0000 to +0.0115; `kd_bank_C` higher in 3/3 seeds).
- Severe (α=0.1): `kd_bank_C` AP is on average 0.0465 higher than `bank_C` (range +0.0123 to +0.0699; `kd_bank_C` higher in 3/3 seeds).

## Paired bootstrap of test AP differences (per seed)

| seed | scenario | a | b | diff | ci_low | ci_high | ci_excludes_zero |
|---|---|---|---|---|---|---|---|
| 42 | iid | kd_committee | ensemble_equal | 0.0024 | -0.0057 | 0.0122 | False |
| 42 | iid | kd_committee | fedavg_global | 0.0275 | 0.0065 | 0.0541 | True |
| 42 | iid | ensemble_equal | fedavg_global | 0.0251 | 0.0043 | 0.0509 | True |
| 42 | iid | kd_bank_A | bank_A | 0.0155 | -0.0046 | 0.0473 | False |
| 42 | iid | kd_bank_B | bank_B | -0.0074 | -0.0181 | 0.0021 | False |
| 42 | iid | kd_bank_C | bank_C | 0.0019 | -0.0062 | 0.0105 | False |
| 42 | mild_non_iid | kd_committee | ensemble_equal | -0.0061 | -0.0164 | 0.0033 | False |
| 42 | mild_non_iid | kd_committee | fedavg_global | 0.0142 | -0.0039 | 0.0369 | False |
| 42 | mild_non_iid | ensemble_equal | fedavg_global | 0.0203 | -0.0004 | 0.0428 | False |
| 42 | mild_non_iid | kd_bank_A | bank_A | -0.0157 | -0.0373 | 0.0004 | False |
| 42 | mild_non_iid | kd_bank_B | bank_B | 0.0120 | -0.0056 | 0.0308 | False |
| 42 | mild_non_iid | kd_bank_C | bank_C | 0.0333 | 0.0052 | 0.0651 | True |
| 42 | moderate_non_iid | kd_committee | ensemble_equal | 0.0077 | -0.0004 | 0.0172 | False |
| 42 | moderate_non_iid | kd_committee | fedavg_global | 0.0408 | 0.0118 | 0.0800 | True |
| 42 | moderate_non_iid | ensemble_equal | fedavg_global | 0.0331 | 0.0015 | 0.0693 | True |
| 42 | moderate_non_iid | kd_bank_A | bank_A | 0.0098 | -0.0312 | 0.0432 | False |
| 42 | moderate_non_iid | kd_bank_B | bank_B | 0.0142 | -0.0007 | 0.0335 | False |
| 42 | moderate_non_iid | kd_bank_C | bank_C | 0.0010 | -0.0097 | 0.0123 | False |
| 42 | severe_non_iid | kd_committee | ensemble_equal | -0.0173 | -0.0451 | 0.0093 | False |
| 42 | severe_non_iid | kd_committee | fedavg_global | -0.0417 | -0.0993 | 0.0122 | False |
| 42 | severe_non_iid | ensemble_equal | fedavg_global | -0.0244 | -0.0646 | 0.0132 | False |
| 42 | severe_non_iid | kd_bank_A | bank_A | 0.0370 | 0.0009 | 0.0786 | True |
| 42 | severe_non_iid | kd_bank_B | bank_B | -0.0275 | -0.0552 | 0.0012 | False |
| 42 | severe_non_iid | kd_bank_C | bank_C | 0.0573 | 0.0214 | 0.0945 | True |
| 123 | iid | kd_committee | ensemble_equal | 0.0108 | 0.0010 | 0.0218 | True |
| 123 | iid | kd_committee | fedavg_global | 0.0778 | 0.0217 | 0.1368 | True |
| 123 | iid | ensemble_equal | fedavg_global | 0.0670 | 0.0100 | 0.1230 | True |
| 123 | iid | kd_bank_A | bank_A | 0.0205 | -0.0054 | 0.0502 | False |
| 123 | iid | kd_bank_B | bank_B | 0.0114 | 0.0001 | 0.0269 | True |
| 123 | iid | kd_bank_C | bank_C | 0.0360 | 0.0092 | 0.0672 | True |
| 123 | mild_non_iid | kd_committee | ensemble_equal | 0.0379 | 0.0053 | 0.0754 | True |
| 123 | mild_non_iid | kd_committee | fedavg_global | 0.0599 | 0.0084 | 0.1136 | True |
| 123 | mild_non_iid | ensemble_equal | fedavg_global | 0.0219 | -0.0355 | 0.0818 | False |
| 123 | mild_non_iid | kd_bank_A | bank_A | 0.0996 | 0.0585 | 0.1442 | True |
| 123 | mild_non_iid | kd_bank_B | bank_B | 0.0155 | -0.0094 | 0.0486 | False |
| 123 | mild_non_iid | kd_bank_C | bank_C | 0.0684 | 0.0252 | 0.1133 | True |
| 123 | moderate_non_iid | kd_committee | ensemble_equal | 0.0053 | -0.0017 | 0.0124 | False |
| 123 | moderate_non_iid | kd_committee | fedavg_global | 0.0935 | 0.0377 | 0.1528 | True |
| 123 | moderate_non_iid | ensemble_equal | fedavg_global | 0.0882 | 0.0345 | 0.1463 | True |
| 123 | moderate_non_iid | kd_bank_A | bank_A | 0.0078 | 0.0001 | 0.0175 | True |
| 123 | moderate_non_iid | kd_bank_B | bank_B | -0.0008 | -0.0087 | 0.0067 | False |
| 123 | moderate_non_iid | kd_bank_C | bank_C | 0.0000 | -0.0276 | 0.0319 | False |
| 123 | severe_non_iid | kd_committee | ensemble_equal | 0.0167 | 0.0025 | 0.0356 | True |
| 123 | severe_non_iid | kd_committee | fedavg_global | 0.0831 | 0.0274 | 0.1417 | True |
| 123 | severe_non_iid | ensemble_equal | fedavg_global | 0.0664 | 0.0115 | 0.1226 | True |
| 123 | severe_non_iid | kd_bank_A | bank_A | -0.0074 | -0.0227 | 0.0013 | False |
| 123 | severe_non_iid | kd_bank_B | bank_B | -0.0170 | -0.0475 | 0.0171 | False |
| 123 | severe_non_iid | kd_bank_C | bank_C | 0.0699 | 0.0187 | 0.1225 | True |
| 2026 | iid | kd_committee | ensemble_equal | -0.0049 | -0.0152 | 0.0026 | False |
| 2026 | iid | kd_committee | fedavg_global | 0.0088 | -0.0201 | 0.0410 | False |
| 2026 | iid | ensemble_equal | fedavg_global | 0.0137 | -0.0166 | 0.0476 | False |
| 2026 | iid | kd_bank_A | bank_A | -0.0079 | -0.0299 | 0.0161 | False |
| 2026 | iid | kd_bank_B | bank_B | -0.0004 | -0.0119 | 0.0096 | False |
| 2026 | iid | kd_bank_C | bank_C | 0.0439 | 0.0086 | 0.0813 | True |
| 2026 | mild_non_iid | kd_committee | ensemble_equal | -0.0035 | -0.0144 | 0.0072 | False |
| 2026 | mild_non_iid | kd_committee | fedavg_global | 0.0123 | -0.0134 | 0.0431 | False |
| 2026 | mild_non_iid | ensemble_equal | fedavg_global | 0.0158 | -0.0115 | 0.0487 | False |
| 2026 | mild_non_iid | kd_bank_A | bank_A | -0.0061 | -0.0173 | 0.0037 | False |
| 2026 | mild_non_iid | kd_bank_B | bank_B | 0.0088 | -0.0091 | 0.0281 | False |
| 2026 | mild_non_iid | kd_bank_C | bank_C | -0.0322 | -0.0535 | -0.0124 | True |
| 2026 | moderate_non_iid | kd_committee | ensemble_equal | 0.0025 | -0.0030 | 0.0088 | False |
| 2026 | moderate_non_iid | kd_committee | fedavg_global | 0.0228 | -0.0002 | 0.0508 | False |
| 2026 | moderate_non_iid | ensemble_equal | fedavg_global | 0.0203 | -0.0035 | 0.0475 | False |
| 2026 | moderate_non_iid | kd_bank_A | bank_A | -0.0174 | -0.0561 | 0.0073 | False |
| 2026 | moderate_non_iid | kd_bank_B | bank_B | 0.0057 | -0.0057 | 0.0209 | False |
| 2026 | moderate_non_iid | kd_bank_C | bank_C | 0.0115 | -0.0125 | 0.0382 | False |
| 2026 | severe_non_iid | kd_committee | ensemble_equal | -0.0167 | -0.0404 | 0.0011 | False |
| 2026 | severe_non_iid | kd_committee | fedavg_global | 0.0142 | -0.0220 | 0.0536 | False |
| 2026 | severe_non_iid | ensemble_equal | fedavg_global | 0.0309 | 0.0056 | 0.0615 | True |
| 2026 | severe_non_iid | kd_bank_A | bank_A | 0.0046 | -0.0517 | 0.0570 | False |
| 2026 | severe_non_iid | kd_bank_B | bank_B | -0.0021 | -0.0083 | 0.0037 | False |
| 2026 | severe_non_iid | kd_bank_C | bank_C | 0.0123 | -0.0038 | 0.0352 | False |

Each interval reflects test-example resampling for one trained pair of models; intervals from different seeds are not independent replications. `ci_excludes_zero` is descriptive: it is not a significance test, no multiple-comparison correction is applied, and it says nothing about variation across training seeds or partitions.

## Ablations (test AP; committee and mean bank model)

| variant | scenario | committee_mean | committee_std | bank_mean | bank_std | n |
|---|---|---|---|---|---|---|
| ensemble:ensemble_val_weighted | iid | 0.8781 | 0.0068 | nan | nan | 3 |
| ensemble:ensemble_val_weighted | moderate_non_iid | 0.8787 | 0.0059 | nan | nan | 3 |
| ensemble:ensemble_val_weighted | severe_non_iid | 0.8618 | 0.0234 | nan | nan | 3 |
| ensemble:primary | iid | 0.8783 | 0.0069 | nan | nan | 3 |
| ensemble:primary | mild_non_iid | 0.8658 | 0.0298 | nan | nan | 3 |
| ensemble:primary | moderate_non_iid | 0.8785 | 0.0063 | nan | nan | 3 |
| ensemble:primary | severe_non_iid | 0.8595 | 0.0240 | nan | nan | 3 |
| kd_full_committee | iid | 0.8836 | 0.0076 | 0.8508 | 0.0049 | 3 |
| kd_full_committee | moderate_non_iid | 0.8845 | 0.0023 | 0.8452 | 0.0257 | 3 |
| kd_full_committee | severe_non_iid | 0.8557 | 0.0258 | 0.8202 | 0.0126 | 3 |
| kd_lambda_0.0 | iid | 0.8802 | 0.0068 | 0.8323 | 0.0017 | 3 |
| kd_lambda_0.0 | moderate_non_iid | 0.8801 | 0.0047 | 0.8415 | 0.0238 | 3 |
| kd_lambda_0.0 | severe_non_iid | 0.8695 | 0.0137 | 0.8136 | 0.0130 | 3 |
| kd_lambda_0.1 | iid | 0.8795 | 0.0072 | 0.8448 | 0.0007 | 3 |
| kd_lambda_0.1 | moderate_non_iid | 0.8839 | 0.0048 | 0.8469 | 0.0286 | 3 |
| kd_lambda_0.1 | severe_non_iid | 0.8600 | 0.0142 | 0.8172 | 0.0234 | 3 |
| kd_lambda_1.0 | iid | 0.8815 | 0.0079 | 0.8518 | 0.0019 | 3 |
| kd_lambda_1.0 | moderate_non_iid | 0.8843 | 0.0049 | 0.8519 | 0.0215 | 3 |
| kd_lambda_1.0 | severe_non_iid | 0.8492 | 0.0347 | 0.8176 | 0.0265 | 3 |
| kd_teacher_val_weighted | iid | 0.8812 | 0.0039 | 0.8490 | 0.0050 | 3 |
| kd_teacher_val_weighted | moderate_non_iid | 0.8843 | 0.0025 | 0.8483 | 0.0222 | 3 |
| kd_teacher_val_weighted | severe_non_iid | 0.8549 | 0.0243 | 0.8195 | 0.0151 | 3 |
| primary | iid | 0.8810 | 0.0055 | 0.8495 | 0.0070 | 3 |
| primary | mild_non_iid | 0.8752 | 0.0052 | 0.8446 | 0.0102 | 3 |
| primary | moderate_non_iid | 0.8836 | 0.0041 | 0.8473 | 0.0234 | 3 |
| primary | severe_non_iid | 0.8537 | 0.0280 | 0.8191 | 0.0153 | 3 |

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
