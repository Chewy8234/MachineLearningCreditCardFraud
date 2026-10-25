## Paired AP comparisons (exact sign-flip permutation; cells = scenario x seed)

| comparison                                 | scope            |   n |   mean_diff |     sd |     min |     max |   n_pos |   p_signflip |   p_holm |
|:-------------------------------------------|:-----------------|----:|------------:|-------:|--------:|--------:|--------:|-------------:|---------:|
| KD committee - lambda0 committee           | all 9 cells      |   9 |     -0.0038 | 0.0137 | -0.0359 |  0.0069 |       5 |       0.5156 |   1      |
| KD committee - lambda0 committee           | iid              |   3 |      0.0008 | 0.0053 | -0.0031 |  0.0069 |       1 |       1      | nan      |
| KD committee - lambda0 committee           | moderate_non_iid |   3 |      0.0035 | 0.0006 |  0.0031 |  0.0042 |       3 |       0.25   | nan      |
| KD committee - lambda0 committee           | severe_non_iid   |   3 |     -0.0159 | 0.0197 | -0.0359 |  0.0036 |       1 |       0.5    | nan      |
| KD committee - ensemble_equal              | all 9 cells      |   9 |      0.0007 | 0.0117 | -0.0173 |  0.0167 |       6 |       0.8633 |   1      |
| KD committee - ensemble_equal              | iid              |   3 |      0.0028 | 0.0079 | -0.0049 |  0.0108 |       2 |       0.75   | nan      |
| KD committee - ensemble_equal              | moderate_non_iid |   3 |      0.0051 | 0.0026 |  0.0025 |  0.0077 |       3 |       0.25   | nan      |
| KD committee - ensemble_equal              | severe_non_iid   |   3 |     -0.0058 | 0.0195 | -0.0173 |  0.0167 |       1 |       0.5    | nan      |
| KD committee - FedAvg MLP                  | all 9 cells      |   9 |      0.0363 | 0.043  | -0.0417 |  0.0935 |       8 |       0.0469 |   0.1562 |
| KD committee - FedAvg MLP                  | iid              |   3 |      0.038  | 0.0357 |  0.0088 |  0.0778 |       3 |       0.25   | nan      |
| KD committee - FedAvg MLP                  | moderate_non_iid |   3 |      0.0524 | 0.0367 |  0.0228 |  0.0935 |       3 |       0.25   | nan      |
| KD committee - FedAvg MLP                  | severe_non_iid   |   3 |      0.0185 | 0.0625 | -0.0417 |  0.0831 |       2 |       0.75   | nan      |
| ensemble_equal - FedAvg MLP                | all 9 cells      |   9 |      0.0356 | 0.0338 | -0.0244 |  0.0882 |       8 |       0.0156 |   0.0781 |
| ensemble_equal - FedAvg MLP                | iid              |   3 |      0.0353 | 0.0281 |  0.0137 |  0.067  |       3 |       0.25   | nan      |
| ensemble_equal - FedAvg MLP                | moderate_non_iid |   3 |      0.0472 | 0.0361 |  0.0203 |  0.0882 |       3 |       0.25   | nan      |
| ensemble_equal - FedAvg MLP                | severe_non_iid   |   3 |      0.0243 | 0.0458 | -0.0244 |  0.0664 |       2 |       0.5    | nan      |
| KD committee - best local (oracle on test) | all 9 cells      |   9 |      0.0113 | 0.0136 | -0.007  |  0.033  |       7 |       0.0391 |   0.1562 |
| KD committee - best local (oracle on test) | iid              |   3 |      0.0174 | 0.0158 |  0.0015 |  0.033  |       3 |       0.25   | nan      |
| KD committee - best local (oracle on test) | moderate_non_iid |   3 |      0.0159 | 0.0106 |  0.0088 |  0.0281 |       3 |       0.25   | nan      |
| KD committee - best local (oracle on test) | severe_non_iid   |   3 |      0.0004 | 0.0107 | -0.007  |  0.0126 |       1 |       1      | nan      |
| KD committee - centralized XGB (ref)       | all 9 cells      |   9 |     -0.0137 | 0.021  | -0.067  |  0.0013 |       1 |       0.0117 |   0.0703 |
| KD committee - centralized XGB (ref)       | iid              |   3 |     -0.0055 | 0.0044 | -0.0105 | -0.002  |       0 |       0.25   | nan      |
| KD committee - centralized XGB (ref)       | moderate_non_iid |   3 |     -0.0029 | 0.0058 | -0.0095 |  0.0013 |       1 |       0.75   | nan      |
| KD committee - centralized XGB (ref)       | severe_non_iid   |   3 |     -0.0328 | 0.0296 | -0.067  | -0.0146 |       0 |       0.25   | nan      |

## Per-bank KD effect (control = lambda 0: same epochs / retraining, no teacher)

| bank   | contrast        |   n |   mean_diff |   n_pos |   p_signflip |   p_holm |
|:-------|:----------------|----:|------------:|--------:|-------------:|---------:|
| A      | KD - local      |   9 |      0.0069 |       6 |       0.2461 |   1      |
| A      | KD - lambda0    |   9 |      0.0069 |       6 |       0.2461 |   1      |
| A      | lambda0 - local |   9 |      0      |       0 |       1      |   1      |
| B      | KD - local      |   9 |     -0.0027 |       3 |       0.5742 |   1      |
| B      | KD - lambda0    |   9 |     -0.0027 |       3 |       0.5742 |   1      |
| B      | lambda0 - local |   9 |      0      |       0 |       1      |   1      |
| C      | KD - local      |   9 |      0.026  |       9 |       0.0039 |   0.0352 |
| C      | KD - lambda0    |   9 |      0.0242 |       8 |       0.0078 |   0.0625 |
| C      | lambda0 - local |   9 |      0.0018 |       3 |       0.832  |   1      |

## KD gain (KD bank - local bank) vs bank size / fraud count (n=27 bank-runs)

Spearman(log10 rows, gain) = 0.494 (p=0.009); Spearman(log10 frauds, gain) = 0.158 (p=0.432).

Per bank/model Spearman (rows vs gain):

- mlp: rho=0.712 (p=0.031, n=9); mean gain=+0.0260
- random_forest: rho=0.644 (p=0.061, n=9); mean gain=-0.0027
- xgboost: rho=-0.068 (p=0.862, n=9); mean gain=+0.0069

Banks with <40 frauds: mean gain +0.0014 (n=5); >=100 frauds: +0.0138 (n=17).

## Seed-level t-intervals: 2/45 exceed AP=1.0 (invalid); logit-scale intervals are bounded and in `bounded_seed_intervals.csv`.

## Ranking stability across seeds (Kendall tau of 6 systems by test AP)

| scenario         | seeds    |   kendall_tau |
|:-----------------|:---------|--------------:|
| iid              | 42-123   |         0.733 |
| iid              | 42-2026  |         0.467 |
| iid              | 123-2026 |         0.467 |
| moderate_non_iid | 42-123   |         0.467 |
| moderate_non_iid | 42-2026  |         0.2   |
| moderate_non_iid | 123-2026 |         0.733 |
| severe_non_iid   | 42-123   |        -0.067 |
| severe_non_iid   | 42-2026  |         0.2   |
| severe_non_iid   | 123-2026 |        -0.333 |

Mean tau = 0.319

## Which system has the highest test AP per (scenario, seed) cell (9 cells)

|                |   n_cells |
|:---------------|----------:|
| kd_committee   |         5 |
| ensemble_equal |         2 |
| fedavg_global  |         1 |
| bank_A         |         1 |
