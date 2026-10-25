# Homogeneous baselines: separating heterogeneity, distillation and ensembling

The main comparison (heterogeneous KD committee vs FedAvg MLP) confounds three things:
*architectural diversity*, *knowledge distillation* and *the strength of the individual
model families* (tree models beat the MLP on ULB). These baselines hold the collaboration
protocol fixed and change only the banks' model families.

## Conditions

| Id | Condition | Profile | Systems (`metrics.csv`) |
|---|---|---|---|
| — | FedAvg MLP, 5 rounds × 2 epochs (existing) | `paper.yaml` | `fedavg_global` |
| — | FedAvg MLP, 5 rounds × 4 epochs (epoch-matched control) | `homog_mlp.yaml`, ablation `fedavg_20_epochs` | `fedavg_global` |
| — | Federated XGBoost, tree bagging | `homog_xgboost.yaml`, method `fedxgb_bagging` | `fedxgb_global` |
| H1 | 3×XGBoost KD committee | `homog_xgboost.yaml` | `kd_committee`, `kd_bank_X` |
| H2 | 3×Random Forest KD committee | `homog_random_forest.yaml` | `kd_committee`, `kd_bank_X` |
| H3 | 3×MLP KD committee | `homog_mlp.yaml` | `kd_committee`, `kd_bank_X` |
| E1–E3 | 3×XGB / 3×RF / 3×MLP local models, equal-weight prediction ensemble, **no KD, no federated training** | same profiles | `ensemble_equal` |
| — | 3×MLP KD with λ = 0 (extra fine-tuning epochs without the teacher) | `homog_mlp.yaml`, ablation `kd_lambda_0.0` | `kd_committee` |
| — | Heterogeneous ensemble / KD committee (proposed), λ = 0 control | `paper.yaml` | `ensemble_equal`, `kd_committee` |

The KD method id is `heterogeneous_kd` for every composition: it names the protocol
(round 0 local training, leave-one-out equal-weight teachers on the ordered proxy set,
λ = 0.5, 5 rounds, final-round selection). Labels say *Homogeneous 3xXGBoost* etc.
E1–E3 are server-side **prediction ensembles** of independently trained local models.
They are not federated training and are labelled as such.

## What is held identical (and how it is enforced)

* `configs/homog_*.yaml` inherit `paper.yaml` (`base_config`). The semantic configuration
  differs only in `clients`, `profile` and, for XGBoost, `fedxgb` (`tests/test_homogeneous.py`).
  So data, split seed and splits, proxy set, preprocessing, partition constraints,
  `models.yaml` hyperparameters, rounds, λ, teacher rule, committee rule, threshold rule
  and model selection are the same.
* Partitions depend only on seed, scenario, constraints and bank names, not on model
  families. `scripts/compare_committees.py` refuses to compare unless every
  (seed, scenario) has the same `partition_sha256` in every profile, the same test labels,
  dataset SHA-256, split bundle and library versions.
* Seeds come from `(run seed, bank, model family)`, so bank A's XGBoost is the same model
  in the heterogeneous and the homogeneous-XGBoost profile. Banks of one homogeneous
  committee get different seeds and different data.
* Each bank reads only its own partition; the proxy file has no labels (the client refuses
  a proxy with labels); the test split is not in the federation view.

## Federated tree baselines

**Federated XGBoost (`fedxgb_bagging`, `heterofl/federation/xgb_bagging.py`).** This is the
tree-bagging scheme of Flower's `FedXgbBagging` strategy. Each round every bank boosts
`fedxgb.trees_per_round` = 20 new trees on its own rows, starting from the current
global model's margin. The server **appends** all new trees; nothing is averaged. With
5 rounds × 3 banks × 20 trees the global model has 300 trees, the size of one local
model. `b0` = logit of the pooled fraud rate (two count scalars per bank). It runs
**in-process** (no Flower transport). Communication is the serialised JSON size of the
trees the protocol would send, so it is an estimate. Banks' trees within a round fit the
same residual independently and can overshoot; that is inherent to the scheme.

**Federated Random Forest: not implemented.** A forest assembled from trees each bank
grows on its own data (`model_matrix.py`'s "federated forest") is mathematically an
equal-weight ensemble of local forests, which is E2. Calling it federated training would
overstate it. A genuinely collaborative horizontal forest needs per-node split statistics
aggregated across banks, i.e. new message types and server logic. That was judged out of
scope for the main study.

## Cost and capacity

| Condition | Model fits per run | Trees / epochs |
|---|---|---|
| E1/E2/E3 | 3 | 3×300 trees / 3×200 trees / 3×10 epochs |
| H1/H2 | 3 × (1 + 5) = 18 (retrained every round) | 18×300 / 18×200 trees |
| H3 | 3 initial + 15 fine-tunes | 3 × (10 + 5×2) epochs |
| Heterogeneous KD | 18 | XGB 6×300, RF 6×200 trees, MLP 20 epochs |
| FedAvg | 15 local updates | 10 epochs (20 in the control) |
| FedXGB | 15 boosting calls | 300 trees in the global model |

Random Forest KD dominates the cost: each round refits 200 trees on the local rows plus
the weighted proxy expansion. On the most quantity-skewed partition (seed 42, α = 0.1,
one bank with 167k rows) that is ≈ 110 s per round with 4 cores per bank. Peak memory is
not measured.

## Commands

```bash
# all four profiles from one source revision, then reports + the paired comparison
scripts/run_homogeneous.sh                          # sequential (12 cores)
CACHE_ROOT=/scratch LANES=2 scripts/run_homogeneous.sh   # two profiles at a time (24 cores)
# or individually (resumable)
python -m heterofl.experiments.runner --config configs/homog_random_forest.yaml
python scripts/compare_committees.py --hetero results/full_v2 \
    --homog-xgboost results/homog_xgboost --homog-random-forest results/homog_random_forest \
    --homog-mlp results/homog_mlp --out results/committee_comparison
```

The heterogeneous reference is re-run as `results/full_v2` with the same source and
environment as the baselines. The legacy `results/full` (older source, dirty tree, no
run identities) is neither modified nor mixed into the comparison.
