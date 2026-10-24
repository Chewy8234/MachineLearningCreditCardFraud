---
tags: [quickstart, tabular, fds]
dataset: [Iris]
framework: [scikit-learn]
---

# Federated Credit Card Fraud Detection

## Reproducing the experiments

**Data:** [Kaggle `mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), `creditcard.csv`
(284,807 transactions, 492 fraud). Place it in the repo root (it is gitignored, 144 MB).
SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`

**Environment:** Python 3.11.0, exact package versions in `quickstart-sklearn/requirements.txt`.

```bash
cd quickstart-sklearn
pip install -r requirements.txt
pip install -e .
python baselines.py   # Exp. 1 local-only + Exp. 2 centralized -> results/

# Exp. 3 homogeneous federated learning, one run per model -> results/federated_<model>.csv
flwr run . --run-config "model='logreg' num-server-rounds=25"    # FedAvg
flwr run . --run-config "model='mlp' num-server-rounds=25"       # FedAvg
flwr run . --run-config "model='xgboost' num-server-rounds=100"  # FedXgbBagging, 1 tree per bank per round
flwr run . --run-config "model='rf' num-server-rounds=1"         # federated forest, 200 trees per bank pooled
```

**Protocol:** rows are shuffled once (seed 42) and split evenly into 3 banks. Each bank is split
70/10/20 into train/validation/test, stratified by `Class`. Models train on train; the fraud
decision threshold is chosen to maximize F1 on validation; test is used only for the reported
scores. Features: V1-V28 plus log(1 + Amount); `Time` is dropped. All randomness uses seed 42,
and reruns give identical results.

---

# Federated Learning with scikit-learn and Flower (Quickstart Example)

This example of Flower uses `scikit-learn`'s `LogisticRegression` model to train a federated learning system on
"iris" (tabular) dataset.
It will help you understand how to adapt Flower for use with `scikit-learn`.
Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to
download, partition and preprocess the dataset.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-sklearn
```

This will create a new directory called `quickstart-sklearn` with the following structure:

```shell
quickstart-sklearn
├── sklearnexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `sklearnexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config penalty="'l1'"
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
