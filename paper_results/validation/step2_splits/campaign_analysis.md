## Split-level paired AP comparisons (cluster = outer split)

| comparison                             | scope            |   n_splits |   n_cells |   mean_diff |   sd_between_splits |   ci_lo |   ci_hi |   n_splits_pos |   p_signflip |   p_holm |
|:---------------------------------------|:-----------------|-----------:|----------:|------------:|--------------------:|--------:|--------:|---------------:|-------------:|---------:|
| KD committee - lambda0 committee       | iid              |         11 |        33 |      0.0003 |              0.0074 | -0.0044 |  0.0037 |              8 |       0.9277 |   1      |
| KD committee - lambda0 committee       | moderate_non_iid |         11 |        33 |      0.0013 |              0.0049 | -0.0016 |  0.0039 |              7 |       0.3906 |   1      |
| KD committee - lambda0 committee       | severe_non_iid   |         11 |        33 |     -0.0161 |              0.0144 | -0.024  | -0.0081 |              2 |       0.0088 |   0.0703 |
| KD committee - lambda0 committee       | pooled           |         11 |        99 |     -0.0048 |              0.005  | -0.0076 | -0.002  |              2 |       0.0107 |   0.0645 |
| KD committee - ensemble_equal          | iid              |         11 |        33 |      0.0011 |              0.008  | -0.0039 |  0.0049 |              8 |       0.6982 |   1      |
| KD committee - ensemble_equal          | moderate_non_iid |         11 |        33 |      0.0051 |              0.0056 |  0.0016 |  0.0079 |             10 |       0.0166 |   0.083  |
| KD committee - ensemble_equal          | severe_non_iid   |         11 |        33 |      0.0116 |              0.0146 |  0.0037 |  0.0199 |              9 |       0.0273 |   0.1641 |
| KD committee - ensemble_equal          | pooled           |         11 |        99 |      0.006  |              0.0056 |  0.003  |  0.0092 |             10 |       0.002  |   0.0176 |
| KD committee - FedAvg MLP              | iid              |         11 |        33 |      0.0715 |              0.0274 |  0.0578 |  0.0884 |             11 |       0.001  |   0.0137 |
| KD committee - FedAvg MLP              | moderate_non_iid |         11 |        33 |      0.0612 |              0.0158 |  0.0528 |  0.0705 |             11 |       0.001  |   0.0137 |
| KD committee - FedAvg MLP              | severe_non_iid   |         11 |        33 |      0.0218 |              0.0196 |  0.011  |  0.0328 |              9 |       0.0059 |   0.0527 |
| KD committee - FedAvg MLP              | pooled           |         11 |        99 |      0.0515 |              0.0168 |  0.0431 |  0.0618 |             11 |       0.001  |   0.0137 |
| ensemble_equal - FedAvg MLP            | iid              |         11 |        33 |      0.0704 |              0.0277 |  0.057  |  0.0875 |             11 |       0.001  |   0.0137 |
| ensemble_equal - FedAvg MLP            | moderate_non_iid |         11 |        33 |      0.0561 |              0.0199 |  0.0458 |  0.0681 |             11 |       0.001  |   0.0137 |
| ensemble_equal - FedAvg MLP            | severe_non_iid   |         11 |        33 |      0.0102 |              0.02   | -0.001  |  0.0212 |              7 |       0.1201 |   0.5127 |
| ensemble_equal - FedAvg MLP            | pooled           |         11 |        99 |      0.0455 |              0.0183 |  0.0365 |  0.0569 |             11 |       0.001  |   0.0137 |
| KD committee - best local (oracle)     | iid              |         11 |        33 |      0.0078 |              0.0115 |  0.0004 |  0.0131 |             10 |       0.0508 |   0.3594 |
| KD committee - best local (oracle)     | moderate_non_iid |         11 |        33 |      0.015  |              0.0097 |  0.0092 |  0.0199 |             10 |       0.0029 |   0.0234 |
| KD committee - best local (oracle)     | severe_non_iid   |         11 |        33 |     -0.0082 |              0.0199 | -0.0202 |  0.0022 |              6 |       0.2266 |   0.5127 |
| KD committee - best local (oracle)     | pooled           |         11 |        99 |      0.0049 |              0.009  | -0.0003 |  0.0098 |              7 |       0.1094 |   0.1514 |
| KD committee - best local (val-chosen) | iid              |         11 |        33 |      0.0089 |              0.0116 |  0.0014 |  0.014  |             10 |       0.0342 |   0.3076 |
| KD committee - best local (val-chosen) | moderate_non_iid |         11 |        33 |      0.019  |              0.0086 |  0.0139 |  0.0238 |             11 |       0.001  |   0.0137 |
| KD committee - best local (val-chosen) | severe_non_iid   |         11 |        33 |     -0.0035 |              0.0239 | -0.0175 |  0.0093 |              7 |       0.6328 |   0.6328 |
| KD committee - best local (val-chosen) | pooled           |         11 |        99 |      0.0081 |              0.0108 |  0.002  |  0.0143 |              7 |       0.0352 |   0.1514 |
| lambda0 committee - ensemble_equal     | iid              |         11 |        33 |      0.0008 |              0.0012 |  0.0002 |  0.0015 |              8 |       0.0449 |   0.3594 |
| lambda0 committee - ensemble_equal     | moderate_non_iid |         11 |        33 |      0.0038 |              0.0033 |  0.002  |  0.0057 |             10 |       0.002  |   0.0195 |
| lambda0 committee - ensemble_equal     | severe_non_iid   |         11 |        33 |      0.0278 |              0.0192 |  0.0174 |  0.0386 |             11 |       0.001  |   0.0137 |
| lambda0 committee - ensemble_equal     | pooled           |         11 |        99 |      0.0108 |              0.0059 |  0.0075 |  0.0141 |             11 |       0.001  |   0.0137 |
| KD bank C(MLP) - lambda0 C             | iid              |         11 |        33 |      0.0173 |              0.0131 |  0.01   |  0.0248 |             10 |       0.002  |   0.0215 |
| KD bank C(MLP) - lambda0 C             | moderate_non_iid |         11 |        33 |      0.0128 |              0.0153 |  0.004  |  0.0215 |             10 |       0.0195 |   0.083  |
| KD bank C(MLP) - lambda0 C             | severe_non_iid   |         11 |        33 |     -0.0117 |              0.0144 | -0.0199 | -0.0039 |              3 |       0.0234 |   0.1641 |
| KD bank C(MLP) - lambda0 C             | pooled           |         11 |        99 |      0.0061 |              0.009  |  0.0009 |  0.0109 |              7 |       0.0508 |   0.1514 |
| KD bank A(XGB) - lambda0 A             | iid              |         11 |        33 |     -0.0014 |              0.0086 | -0.0065 |  0.003  |              6 |       0.6074 |   1      |
| KD bank A(XGB) - lambda0 A             | moderate_non_iid |         11 |        33 |      0.0106 |              0.0095 |  0.0051 |  0.0159 |             10 |       0.0068 |   0.0479 |
| KD bank A(XGB) - lambda0 A             | severe_non_iid   |         11 |        33 |      0.0237 |              0.0174 |  0.0147 |  0.0343 |             11 |       0.001  |   0.0137 |
| KD bank A(XGB) - lambda0 A             | pooled           |         11 |        99 |      0.011  |              0.009  |  0.0061 |  0.0162 |             10 |       0.002  |   0.0176 |
| KD bank B(RF) - lambda0 B              | iid              |         11 |        33 |      0.0025 |              0.0044 | -0.0001 |  0.0049 |              7 |       0.0957 |   0.5742 |
| KD bank B(RF) - lambda0 B              | moderate_non_iid |         11 |        33 |      0.0033 |              0.011  | -0.0025 |  0.0098 |              7 |       0.3799 |   1      |
| KD bank B(RF) - lambda0 B              | severe_non_iid   |         11 |        33 |      0.0062 |              0.0112 | -0.0002 |  0.0123 |              7 |       0.1025 |   0.5127 |
| KD bank B(RF) - lambda0 B              | pooled           |         11 |        99 |      0.004  |              0.0051 |  0.0012 |  0.007  |              8 |       0.0303 |   0.1514 |
| lambda0 bank C(MLP) - local C          | iid              |         11 |        33 |      0.0103 |              0.0139 |  0.0025 |  0.0182 |              9 |       0.0293 |   0.293  |
| lambda0 bank C(MLP) - local C          | moderate_non_iid |         11 |        33 |      0.0252 |              0.0159 |  0.0159 |  0.0337 |             10 |       0.002  |   0.0195 |
| lambda0 bank C(MLP) - local C          | severe_non_iid   |         11 |        33 |      0.0354 |              0.0221 |  0.0239 |  0.0486 |             11 |       0.001  |   0.0137 |
| lambda0 bank C(MLP) - local C          | pooled           |         11 |        99 |      0.0236 |              0.0106 |  0.0175 |  0.0294 |             11 |       0.001  |   0.0137 |
| KD bank C(MLP) - local C               | iid              |         11 |        33 |      0.0277 |              0.0168 |  0.0196 |  0.0379 |             11 |       0.001  |   0.0137 |
| KD bank C(MLP) - local C               | moderate_non_iid |         11 |        33 |      0.038  |              0.0174 |  0.028  |  0.0477 |             11 |       0.001  |   0.0137 |
| KD bank C(MLP) - local C               | severe_non_iid   |         11 |        33 |      0.0237 |              0.0261 |  0.0109 |  0.0399 |              9 |       0.0039 |   0.0391 |
| KD bank C(MLP) - local C               | pooled           |         11 |        99 |      0.0298 |              0.0116 |  0.0234 |  0.0364 |             11 |       0.001  |   0.0137 |
| KD bank A(XGB) - local A               | iid              |         11 |        33 |     -0.0014 |              0.0086 | -0.0065 |  0.003  |              6 |       0.6074 |   1      |
| KD bank A(XGB) - local A               | moderate_non_iid |         11 |        33 |      0.0106 |              0.0095 |  0.0051 |  0.0159 |             10 |       0.0068 |   0.0479 |
| KD bank A(XGB) - local A               | severe_non_iid   |         11 |        33 |      0.0237 |              0.0174 |  0.0147 |  0.0343 |             11 |       0.001  |   0.0137 |
| KD bank A(XGB) - local A               | pooled           |         11 |        99 |      0.011  |              0.009  |  0.0061 |  0.0162 |             10 |       0.002  |   0.0176 |
| KD bank B(RF) - local B                | iid              |         11 |        33 |      0.0025 |              0.0044 | -0.0001 |  0.0049 |              7 |       0.0957 |   0.5742 |
| KD bank B(RF) - local B                | moderate_non_iid |         11 |        33 |      0.0033 |              0.011  | -0.0025 |  0.0098 |              7 |       0.3799 |   1      |
| KD bank B(RF) - local B                | severe_non_iid   |         11 |        33 |      0.0062 |              0.0112 | -0.0002 |  0.0123 |              7 |       0.1025 |   0.5127 |
| KD bank B(RF) - local B                | pooled           |         11 |        99 |      0.004  |              0.0051 |  0.0012 |  0.007  |              8 |       0.0303 |   0.1514 |

## Mixed model `diff ~ 1 + (1|split)` over all cells

| comparison                       |   estimate |     se |       z |      p |   n_splits |   n_cells |   p_holm |
|:---------------------------------|-----------:|-------:|--------:|-------:|-----------:|----------:|---------:|
| KD committee - lambda0 committee |    -0.0048 | 0.0025 | -1.966  | 0.0493 |         11 |        99 |   0.0493 |
| KD committee - ensemble_equal    |     0.006  | 0.0019 |  3.1103 | 0.0019 |         11 |        99 |   0.0037 |
| KD committee - FedAvg MLP        |     0.0515 | 0.0055 |  9.3161 | 0      |         11 |        99 |   0      |
| ensemble_equal - FedAvg MLP      |     0.0455 | 0.0062 |  7.3526 | 0      |         11 |        99 |   0      |

## Mean test AP by system (over all cells)

|                                                   |   mean |    std |   count |
|:--------------------------------------------------|-------:|-------:|--------:|
| ('iid', 'bank_A')                                 | 0.8352 | 0.0303 |      33 |
| ('iid', 'bank_B')                                 | 0.8159 | 0.0342 |      33 |
| ('iid', 'bank_C')                                 | 0.7353 | 0.0467 |      33 |
| ('iid', 'centralized_mlp')                        | 0.7902 | 0.0491 |      33 |
| ('iid', 'centralized_random_forest')              | 0.8443 | 0.0277 |      33 |
| ('iid', 'centralized_xgboost')                    | 0.855  | 0.0318 |      33 |
| ('iid', 'ensemble_equal')                         | 0.8431 | 0.0311 |      33 |
| ('iid', 'fedavg_global')                          | 0.7727 | 0.0494 |      33 |
| ('iid', 'kd_bank_A')                              | 0.8339 | 0.0294 |      33 |
| ('iid', 'kd_bank_A_round0')                       | 0.8352 | 0.0303 |      33 |
| ('iid', 'kd_bank_B')                              | 0.8184 | 0.0321 |      33 |
| ('iid', 'kd_bank_B_round0')                       | 0.8159 | 0.0342 |      33 |
| ('iid', 'kd_bank_C')                              | 0.7629 | 0.0565 |      33 |
| ('iid', 'kd_bank_C_round0')                       | 0.7353 | 0.0467 |      33 |
| ('iid', 'kd_committee')                           | 0.8442 | 0.0303 |      33 |
| ('moderate_non_iid', 'bank_A')                    | 0.8017 | 0.06   |      33 |
| ('moderate_non_iid', 'bank_B')                    | 0.808  | 0.035  |      33 |
| ('moderate_non_iid', 'bank_C')                    | 0.7466 | 0.0482 |      33 |
| ('moderate_non_iid', 'centralized_mlp')           | 0.7902 | 0.0491 |      33 |
| ('moderate_non_iid', 'centralized_random_forest') | 0.8443 | 0.0277 |      33 |
| ('moderate_non_iid', 'centralized_xgboost')       | 0.855  | 0.0318 |      33 |
| ('moderate_non_iid', 'ensemble_equal')            | 0.837  | 0.0304 |      33 |
| ('moderate_non_iid', 'fedavg_global')             | 0.7809 | 0.0455 |      33 |
| ('moderate_non_iid', 'kd_bank_A')                 | 0.8123 | 0.0526 |      33 |
| ('moderate_non_iid', 'kd_bank_A_round0')          | 0.8017 | 0.06   |      33 |
| ('moderate_non_iid', 'kd_bank_B')                 | 0.8113 | 0.0384 |      33 |
| ('moderate_non_iid', 'kd_bank_B_round0')          | 0.808  | 0.035  |      33 |
| ('moderate_non_iid', 'kd_bank_C')                 | 0.7845 | 0.049  |      33 |
| ('moderate_non_iid', 'kd_bank_C_round0')          | 0.7466 | 0.0482 |      33 |
| ('moderate_non_iid', 'kd_committee')              | 0.8422 | 0.0321 |      33 |
| ('severe_non_iid', 'bank_A')                      | 0.7754 | 0.0672 |      33 |
| ('severe_non_iid', 'bank_B')                      | 0.7454 | 0.0742 |      33 |
| ('severe_non_iid', 'bank_C')                      | 0.7444 | 0.0487 |      33 |
| ('severe_non_iid', 'centralized_mlp')             | 0.7902 | 0.0491 |      33 |
| ('severe_non_iid', 'centralized_random_forest')   | 0.8443 | 0.0277 |      33 |
| ('severe_non_iid', 'centralized_xgboost')         | 0.855  | 0.0318 |      33 |
| ('severe_non_iid', 'ensemble_equal')              | 0.79   | 0.0742 |      33 |
| ('severe_non_iid', 'fedavg_global')               | 0.7798 | 0.0526 |      33 |
| ('severe_non_iid', 'kd_bank_A')                   | 0.7991 | 0.0582 |      33 |
| ('severe_non_iid', 'kd_bank_A_round0')            | 0.7754 | 0.0672 |      33 |
| ('severe_non_iid', 'kd_bank_B')                   | 0.7516 | 0.0707 |      33 |
| ('severe_non_iid', 'kd_bank_B_round0')            | 0.7454 | 0.0742 |      33 |
| ('severe_non_iid', 'kd_bank_C')                   | 0.7681 | 0.0473 |      33 |
| ('severe_non_iid', 'kd_bank_C_round0')            | 0.7444 | 0.0487 |      33 |
| ('severe_non_iid', 'kd_committee')                | 0.8016 | 0.0655 |      33 |
