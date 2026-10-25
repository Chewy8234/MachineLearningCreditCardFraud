## Calibration, mean over all default-hyperparameter cells

|                     |      ap |   brier |   logloss |     ece |   mean_pred |   prevalence |
|:--------------------|--------:|--------:|----------:|--------:|------------:|-------------:|
| ('kd', 'kd_bank_A') | 0.81643 | 0.00049 |   0.00317 | 0.00047 |     0.00171 |      0.00172 |
| ('kd', 'kd_bank_B') | 0.7954  | 0.00059 |   0.00455 | 0.00095 |     0.00231 |      0.00172 |
| ('kd', 'kd_bank_C') | 0.77337 | 0.00054 |   0.00335 | 0.00044 |     0.00174 |      0.00172 |
| ('local', 'bank_A') | 0.80518 | 0.00055 |   0.00365 | 0.00042 |     0.00153 |      0.00172 |
| ('local', 'bank_B') | 0.79124 | 0.00065 |   0.00511 | 0.00142 |     0.00271 |      0.00172 |
| ('local', 'bank_C') | 0.74382 | 0.0008  |   0.00406 | 0.00083 |     0.00126 |      0.00172 |

## Combiners: test AP; diff vs paper mean (all 102 cells x {local, kd})

|                         |     ap |   diff_vs_mean |   frac_cells_better |
|:------------------------|-------:|---------------:|--------------------:|
| ('kd', 'geo_mean')      | 0.8318 |         0.0011 |              0.6569 |
| ('kd', 'logit_mean')    | 0.8291 |        -0.0015 |              0.6373 |
| ('kd', 'max')           | 0.8057 |        -0.025  |              0.1373 |
| ('kd', 'mean')          | 0.8307 |         0      |              0      |
| ('kd', 'platt_mean')    | 0.8293 |        -0.0014 |              0.5294 |
| ('kd', 'rank_mean')     | 0.8239 |        -0.0068 |              0.2745 |
| ('kd', 'stacked_lr')    | 0.8375 |         0.0069 |              0.5882 |
| ('local', 'geo_mean')   | 0.8252 |         0.0005 |              0.5784 |
| ('local', 'logit_mean') | 0.8237 |        -0.001  |              0.6275 |
| ('local', 'max')        | 0.8093 |        -0.0154 |              0.2451 |
| ('local', 'mean')       | 0.8247 |         0      |              0      |
| ('local', 'platt_mean') | 0.823  |        -0.0017 |              0.5588 |
| ('local', 'rank_mean')  | 0.8148 |        -0.0099 |              0.1667 |
| ('local', 'stacked_lr') | 0.8299 |         0.0052 |              0.5784 |

## Cost / alert budget (mean over cells, default hyperparameters; lower cost better)

|                                    |   precision@50 |   precision@100 |   precision@200 |   cost_fn100_valthr |   cost_fn100_oracle |   cost_fn100_all_negative |   cost_fn500_valthr |   cost_fn500_oracle |   cost_fn500_all_negative |
|:-----------------------------------|---------------:|----------------:|----------------:|--------------------:|--------------------:|--------------------------:|--------------------:|--------------------:|--------------------------:|
| ('kd', 'best_single_by_val(A)')    |            1   |             0.8 |             0.4 |              1515.4 |              1402.6 |                      9800 |              6569.4 |              5771.5 |                     49000 |
| ('kd', 'best_single_by_val(B)')    |            1   |             0.8 |             0.4 |              1542.9 |              1462.9 |                      9800 |              6812.5 |              6219.1 |                     49000 |
| ('kd', 'best_single_by_val(C)')    |            0.9 |             0.8 |             0.4 |              1492.5 |              1334.9 |                      9800 |              6512.9 |              5505.2 |                     49000 |
| ('kd', 'committee_mean')           |            1   |             0.8 |             0.4 |              1467.9 |              1358.4 |                      9800 |              6365.1 |              5485.6 |                     49000 |
| ('local', 'best_single_by_val(A)') |            1   |             0.8 |             0.4 |              1527.8 |              1420.5 |                      9800 |              6584.1 |              5866.8 |                     49000 |
| ('local', 'best_single_by_val(B)') |            1   |             0.8 |             0.4 |              1519.5 |              1467   |                      9800 |              6971.7 |              6521.9 |                     49000 |
| ('local', 'best_single_by_val(C)') |            0.9 |             0.8 |             0.4 |              1545.8 |              1415.9 |                      9800 |              6810.7 |              5820.9 |                     49000 |
| ('local', 'committee_mean')        |            1   |             0.8 |             0.4 |              1493.2 |              1389   |                      9800 |              6485.3 |              5571   |                     49000 |

## AP on test rows with / without an exact training twin (split 42 paper group)

|                                               |     n |   n_fraud |     ap |
|:----------------------------------------------|------:|----------:|-------:|
| ('kd_bank_A', 'no_train_duplicate')           | 56663 |        93 | 0.8466 |
| ('kd_bank_A', 'with_train_duplicate')         |   299 |         5 | 1      |
| ('kd_bank_B', 'no_train_duplicate')           | 56663 |        93 | 0.8249 |
| ('kd_bank_B', 'with_train_duplicate')         |   299 |         5 | 1      |
| ('kd_bank_C', 'no_train_duplicate')           | 56663 |        93 | 0.819  |
| ('kd_bank_C', 'with_train_duplicate')         |   299 |         5 | 0.9536 |
| ('kd_committee_mean', 'no_train_duplicate')   | 56663 |        93 | 0.8664 |
| ('kd_committee_mean', 'with_train_duplicate') |   299 |         5 | 1      |
