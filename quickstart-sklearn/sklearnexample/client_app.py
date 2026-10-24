"""sklearnexample: A Flower / sklearn app."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import average_precision_score, log_loss, recall_score

from sklearnexample import committee
from sklearnexample.models import load_global, train_local
from sklearnexample.task import UNIQUE_LABELS, best_threshold, compute_metrics, load_data

# Flower ClientApp
app = ClientApp()


def _load_bank_data(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    return load_data(partition_id, num_partitions, context.run_config["data-path"])


def _train_committee(ndarrays, context: Context):
    """Experiment 4: train this bank's own model type with distilled targets."""
    X_train, y_train, _, _, _, _ = _load_bank_data(context)
    name = committee.bank_model(context.node_config["partition-id"], context.run_config)
    targets = committee.soft_targets(
        y_train, ndarrays[0], float(context.run_config["kd-alpha"])
    )
    model = committee.fit_soft(name, X_train, y_train, targets)
    # Keep the model on this bank for evaluation; it never leaves the bank
    context.state["committee-model"] = committee.dump_model(model)
    # Class averages come from training data only; validation is kept for
    # choosing thresholds and the reported round
    stats = committee.class_logit_stats(model, X_train, y_train)
    return stats, X_train, y_train, lambda X: model.predict_proba(X)[:, 1]


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    model_name = context.run_config["model"]
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    if model_name == "committee":
        ndarrays, X_train, y_train, predict_proba = _train_committee(ndarrays, context)
    else:
        X_train, y_train, _, _, _, _ = _load_bank_data(context)
        # Start from the global model and train on this bank's data
        ndarrays, predict_proba = train_local(
            model_name,
            ndarrays,
            X_train,
            y_train,
            context.run_config,
            msg.content["config"]["server-round"],
        )

    # Let's compute train loss
    y_train_proba = predict_proba(X_train)
    train_logloss = log_loss(y_train, y_train_proba, labels=UNIQUE_LABELS)
    train_recall = recall_score(y_train, y_train_proba >= 0.5, zero_division=0)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_train),
        "bank": context.node_config["partition-id"],
        "train_logloss": train_logloss,
        "train_recall": train_recall,
    }
    content = RecordDict(
        {"arrays": ArrayRecord(ndarrays), "metrics": MetricRecord(metrics)}
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model_name = context.run_config["model"]
    _, _, X_val, y_val, X_test, y_test = _load_bank_data(context)

    if model_name == "committee":
        # Each bank is scored with its own model
        model = committee.load_model(context.state["committee-model"])
        predict_proba = lambda X: model.predict_proba(X)[:, 1]
    else:
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        predict_proba = load_global(model_name, ndarrays, context.run_config)

    y_val_proba = predict_proba(X_val)
    y_test_proba = predict_proba(X_test)
    loss = log_loss(y_test, y_test_proba, labels=UNIQUE_LABELS)
    # Each bank tunes its fraud threshold on its own validation set
    threshold = best_threshold(y_val, y_val_proba)
    # Accuracy is misleading with ~0.17% fraud, so report fraud-focused metrics
    metrics = compute_metrics(y_test, y_test_proba, threshold)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_test),
        "bank": context.node_config["partition-id"],
        "test_logloss": loss,
        # Validation PR-AUC picks which round to report; test stays unseen
        "val_pr_auc": average_precision_score(y_val, y_val_proba),
        **metrics,
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
