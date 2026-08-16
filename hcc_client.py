"""
hcc_client.py

Week 2, Phase 1 -- Real HCC federated client.

Federates SAE + GRU only (Keras models whose weights FedAvg can average).
DBN (BernoulliRBM + LogisticRegression, from models.py) is trained locally
per client, per round -- it is NOT federated in this version. This is a
documented scope decision for Phase 1 (see federated_models.py docstring
and the thesis execution plan): DBN has no gradient-based weights that
FedAvg's weighted-averaging can meaningfully combine across clients.

Each simulated client gets its own private shard of the 132 real training
samples (see partition_clients.py for how the shards are built). The test
set (X_test/y_test, held out by data_loader.load_data()) is shared across
clients for evaluation ONLY -- this is standard practice for measuring the
federated model's generalization and is not a leak, since no client's
fit() ever sees it.
"""

import numpy as np
import flwr as fl

from federated_models import (
    build_sae, build_gru,
    local_finetune_sae, local_finetune_gru,
)
from models import train_dbn


def get_combined_weights(autoencoder, gru_model):
    """Flatten [SAE weights..., GRU weights...] into one list for Flower."""
    return autoencoder.get_weights() + gru_model.get_weights()


def set_combined_weights(autoencoder, gru_model, parameters):
    """Inverse of get_combined_weights: split and load weights back in."""
    n_sae = len(autoencoder.get_weights())
    autoencoder.set_weights(parameters[:n_sae])
    gru_model.set_weights(parameters[n_sae:])


class HCCClient(fl.client.NumPyClient):
    def __init__(self, client_id, X_train, y_train, X_test, y_test,
                 encoding_dim=32, local_epochs=5):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.encoding_dim = encoding_dim
        self.local_epochs = local_epochs
        self.input_dim = X_train.shape[1]

        # Build (untrained) architectures once; weights get overwritten by
        # the server's global parameters at the start of every fit()/evaluate().
        self.autoencoder, self.encoder = build_sae(self.input_dim, encoding_dim)
        self.gru = build_gru(encoding_dim)

        # Local-only: never sent to the server, never federated.
        self.local_dbn = None

    def get_parameters(self, config):
        return get_combined_weights(self.autoencoder, self.gru)

    def fit(self, parameters, config):
        set_combined_weights(self.autoencoder, self.gru, parameters)

        # A FEW local epochs per round (fine-tuning from the current global
        # weights) -- not a full from-scratch 3000-epoch train. This is what
        # makes it "federated" rather than "independent local training
        # averaged after the fact".
        local_finetune_sae(self.autoencoder, self.X_train, local_epochs=self.local_epochs)
        X_train_encoded = self.encoder.predict(self.X_train, verbose=0)
        local_finetune_gru(self.gru, X_train_encoded, self.y_train, local_epochs=self.local_epochs)

        # Fresh local-only DBN each round, trained on this client's own
        # SAE-encoded data (mirrors the centralized pipeline's SAE -> DBN
        # flow), but never sent to the server.
        self.local_dbn = train_dbn(X_train_encoded, self.y_train)

        num_examples = len(self.X_train)
        return get_combined_weights(self.autoencoder, self.gru), num_examples, {}

    def evaluate(self, parameters, config):
        set_combined_weights(self.autoencoder, self.gru, parameters)

        X_test_encoded = self.encoder.predict(self.X_test, verbose=0)
        X_test_reshaped = X_test_encoded.reshape(X_test_encoded.shape[0], 1, self.encoding_dim)
        gru_proba = self.gru.predict(X_test_reshaped, verbose=0).flatten()

        # Metric reported to Flower for aggregation is GRU-only (the
        # federated component). DBN's local contribution is intentionally
        # NOT part of the federated aggregation this phase.
        eps = 1e-7
        p = np.clip(gru_proba, eps, 1 - eps)
        loss = float(np.mean(-(self.y_test * np.log(p) + (1 - self.y_test) * np.log(1 - p))))
        gru_preds = (gru_proba >= 0.5).astype(int)
        accuracy = float(np.mean(gru_preds == self.y_test))

        print(f"[Client {self.client_id}] evaluate() -> GRU-only loss={loss:.4f}, acc={accuracy:.4f}")

        num_examples = len(self.X_test)
        return loss, num_examples, {"accuracy": accuracy}
