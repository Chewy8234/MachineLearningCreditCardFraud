"""sklearnexample: A Flower / sklearn app."""

from pathlib import Path

import pandas as pd
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from sklearnexample.models import initial_arrays, make_strategy, save_global

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    model_name: str = context.run_config["model"]
    min_nodes: int = context.run_config["min-available-clients"]

    # Global model the banks start from
    arrays = ArrayRecord(initial_arrays(model_name, context.run_config["penalty"]))

    # FedAvg for weight-based models, tree-merging strategies for XGBoost and RF
    strategy = make_strategy(model_name, min_nodes)

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model parameters
    print("\nSaving final model to disk...")
    path = save_global(model_name, result.arrays.to_numpy_ndarrays())
    print(f"Saved {path}")

    # Test metrics per round (averaged over banks, weighted by test size)
    rows = [
        {"round": rnd, **dict(metrics)}
        for rnd, metrics in result.evaluate_metrics_clientapp.items()
    ]
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"federated_{model_name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")
