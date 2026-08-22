"""
run_federated_hcc.py

Week 2-4, Phase 1 -- entry point. Runs the REAL SAE+GRU federated pipeline
(2 clients, IID split by default) on the real 132-sample HCC training set,
using Flower's FedAvg simulation.

DBN is trained locally per client (not federated) -- see hcc_client.py.
Federated evaluation metrics (loss/accuracy) reflect the GRU-only pipeline;
this is a documented scope decision for Phase 1.

Week 4 additions:
  - DP_ENABLED / CLIP_NORM / NOISE_MULTIPLIER: toggle differential privacy
    (clip + noise) on the client-side updates. DP_ENABLED=False reproduces
    Week 2/3 exactly (the "no-DP control" run for comparison).
  - Per-round communication overhead (bytes sent) and local training time
    are collected from each client's fit() metrics and written to a CSV
    (results/fl_metrics.csv) so multiple DP settings can be compared later.

Usage:
    python run_federated_hcc.py
"""

import csv
import os

import flwr as fl
from flwr.server.strategy import FedAvg

from data_loader import load_data
from federated_models import build_sae, build_gru
from hcc_client import HCCClient, get_combined_weights
from partition_clients import split_iid, split_non_iid, print_split_summary


NUM_CLIENTS = 2
NUM_ROUNDS = 10
LOCAL_EPOCHS = 5
ENCODING_DIM = 32
USE_NON_IID = os.environ.get("USE_NON_IID", "0") == "1"

# --- Differential privacy (Week 4) ---
# DP_ENABLED=False -> exactly Week 2/3 behavior (no-DP control run).
# When True, each client clips its local update to CLIP_NORM and adds
# Gaussian noise with std = NOISE_MULTIPLIER * CLIP_NORM before sending it
# to the server. See dp_utils.py for the mechanism and its scope (no formal
# epsilon accounting yet -- report results in terms of these two knobs).
#
# All three can be overridden via environment variables, so a shell loop
# can sweep settings without editing this file each time, e.g.:
#   DP_ENABLED=1 NOISE_MULTIPLIER=0.3 python run_federated_hcc.py
DP_ENABLED = os.environ.get("DP_ENABLED", "0") == "1"
CLIP_NORM = float(os.environ.get("CLIP_NORM", "1.0"))
NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", "0.1"))

RESULTS_CSV = "results/fl_metrics.csv"


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_metrics_aggregation(metrics):
    """Aggregate each client's fit()-time metrics (train_time_sec,
    bytes_sent, and pre_clip_update_norm if DP is on) via weighted average,
    weighted by each client's number of local training examples."""
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)

    keys = metrics[0][1].keys()
    aggregated = {}
    for key in keys:
        weighted_sum = sum(num_examples * m[key] for num_examples, m in metrics)
        aggregated[key] = weighted_sum / total_examples
    # Also report the total (not just average) bytes sent this round --
    # more directly meaningful as "communication overhead per round".
    aggregated["bytes_sent_total"] = sum(m["bytes_sent"] for _, m in metrics)
    return aggregated



def fit_config(server_round: int):
    """Passed to each client's fit() via `config` so the client knows the
    true Flower round number (its own internal state does not reliably
    persist across rounds in the simulation)."""
    return {"server_round": server_round}

def main():
    X_train, X_test, y_train, y_test, selected_features, _ = load_data()

    if USE_NON_IID:
        shards = split_non_iid(X_train, y_train)
        print_split_summary(shards, "Non-IID")
    else:
        shards = split_iid(X_train, y_train, num_clients=NUM_CLIENTS)
        print_split_summary(shards, "IID")

    input_dim = X_train.shape[1]

    def client_fn(context: fl.common.Context):
        cid_int = int(context.node_config["partition-id"])
        X_c, y_c = shards[cid_int]
        # NOTE: all clients are evaluated on the SAME held-out global test
        # set (X_test/y_test) -- standard practice for measuring the
        # FEDERATED MODEL's generalization. Not a leak: the test set is
        # never touched by any client's fit().
        return HCCClient(
            client_id=cid_int,
            X_train=X_c, y_train=y_c,
            X_test=X_test, y_test=y_test,
            encoding_dim=ENCODING_DIM,
            local_epochs=LOCAL_EPOCHS,
            dp_enabled=DP_ENABLED,
            clip_norm=CLIP_NORM,
            noise_multiplier=NOISE_MULTIPLIER,
        ).to_client()

    # Build a throwaway pair of models just to get correctly-shaped initial
    # global parameters (random Keras init) for the strategy to start from.
    init_autoencoder, _ = build_sae(input_dim, ENCODING_DIM)
    init_gru = build_gru(ENCODING_DIM)
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_combined_weights(init_autoencoder, init_gru)
    )

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\n=== Federated HCC (SAE+GRU) simulation finished ===")
    print("Aggregated evaluate results per round:", history.metrics_distributed)
    print("Aggregated fit metrics per round:", history.metrics_distributed_fit)

    write_results_csv(history)


def write_results_csv(history):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    loss_by_round = dict(history.losses_distributed)
    acc_by_round = dict(history.metrics_distributed.get("accuracy", []))
    time_by_round = dict(history.metrics_distributed_fit.get("train_time_sec", []))
    bytes_by_round = dict(history.metrics_distributed_fit.get("bytes_sent_total", []))
    clip_by_round = dict(history.metrics_distributed_fit.get("pre_clip_update_norm", []))

    rounds = sorted(loss_by_round.keys())
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "non_iid", "dp_enabled", "clip_norm", "noise_multiplier",
                "round", "loss", "accuracy", "avg_train_time_sec",
                "total_bytes_sent", "avg_pre_clip_update_norm",
            ])
        for r in rounds:
            writer.writerow([
                USE_NON_IID, DP_ENABLED, CLIP_NORM, NOISE_MULTIPLIER,
                r, loss_by_round.get(r), acc_by_round.get(r),
                time_by_round.get(r), bytes_by_round.get(r),
                clip_by_round.get(r),
            ])
    print(f"\nResults appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
