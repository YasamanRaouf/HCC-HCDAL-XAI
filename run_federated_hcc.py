"""
run_federated_hcc.py

Week 2, Phase 1 -- entry point. Runs the REAL SAE+GRU federated pipeline
(2 clients, IID split by default) on the real 132-sample HCC training set,
using Flower's FedAvg simulation.

DBN is trained locally per client (not federated) -- see hcc_client.py.
Federated evaluation metrics (loss/accuracy) reflect the GRU-only pipeline;
this is a documented scope decision for Phase 1.

Usage:
    python run_federated_hcc.py
"""

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
USE_NON_IID = True  # flip to True once the IID run is confirmed working


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


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


if __name__ == "__main__":
    main()
