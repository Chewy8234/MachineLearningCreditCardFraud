"""sklearnexample: A Flower / sklearn app."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import log_loss, recall_score

from sklearnexample.models import load_global, train_local
from sklearnexample.task import UNIQUE_LABELS, best_threshold, compute_metrics, load_data

# Flower ClientApp
app = ClientApp()


def _load_bank_data(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    return load_data(partition_id, num_partitions, context.run_config["data-path"])


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    model_name = context.run_config["model"]
    X_train, y_train, _, _, _, _ = _load_bank_data(context)

    # Start from the global model and train on this bank's data
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    ndarrays, predict_proba = train_local(
        model_name, ndarrays, X_train, y_train, context.run_config
    )

    # Let's compute train loss
    y_train_proba = predict_proba(X_train)
    train_logloss = log_loss(y_train, y_train_proba, labels=UNIQUE_LABELS)
    train_recall = recall_score(y_train, y_train_proba >= 0.5, zero_division=0)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_train),
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

    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    predict_proba = load_global(model_name, ndarrays, context.run_config)

    y_test_proba = predict_proba(X_test)
    loss = log_loss(y_test, y_test_proba, labels=UNIQUE_LABELS)
    # Each bank tunes its fraud threshold on its own validation set
    threshold = best_threshold(y_val, predict_proba(X_val))
    # Accuracy is misleading with ~0.17% fraud, so report fraud-focused metrics
    metrics = compute_metrics(y_test, y_test_proba, threshold)

    # Construct and return reply Message
    metrics = {"num-examples": len(X_test), "test_logloss": loss, **metrics}
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
