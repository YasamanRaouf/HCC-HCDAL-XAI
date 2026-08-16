"""
federated_models.py

Model *architecture* builders for the federated (Flower) version of the
SAE + GRU pipeline. These mirror the exact architecture/hyperparameters of
train_sae/train_gru in models.py (same layers, activations, regularizer,
dropout), but are split into "build" vs "train" so that a Flower client can:

  1. build the model,
  2. load in a given set of weights (the current global federated weights),
  3. fine-tune locally for a SMALL number of epochs (a "round" of local
     training, NOT a full from-scratch 3000-epoch train), and
  4. hand the updated weights back to the server for aggregation.

This is a deliberate, documented deviation from calling train_sae/train_gru
directly: those functions always build a FRESH model and train it fully
from scratch (up to 3000 epochs), which would silently defeat the purpose
of federated learning -- there would be nothing meaningful to aggregate
across rounds, since each client would just be re-solving its own local
problem independently every round instead of continuing from the shared
global weights.

DBN (BernoulliRBM + LogisticRegression, from models.py) is NOT federated in
this version -- it has no gradient-based weights that FedAvg can average --
so each client trains and keeps its own local DBN. This is documented as a
known, intentional scope decision for Phase 1 (see the thesis execution
plan and hcc_client.py).
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras.regularizers import l1


def build_sae(input_dim: int, encoding_dim: int = 32):
    """Same architecture as train_sae() in models.py, but NOT trained yet."""
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu", activity_regularizer=l1(1e-4))(input_layer)
    decoded = Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def build_gru(input_dim: int):
    """Same architecture as train_gru() in models.py, but NOT trained yet."""
    model = tf.keras.Sequential([
        GRU(64, input_shape=(1, input_dim)),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def local_finetune_sae(autoencoder, X_train, local_epochs: int = 5, batch_size: int = 16):
    autoencoder.fit(
        X_train, X_train,
        epochs=local_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    return autoencoder


def local_finetune_gru(model, X_train_encoded, y_train, local_epochs: int = 5, batch_size: int = 16):
    X_reshaped = X_train_encoded.reshape(X_train_encoded.shape[0], 1, X_train_encoded.shape[1])
    model.fit(
        X_reshaped, y_train,
        epochs=local_epochs,
        batch_size=batch_size,
        verbose=0,
    )
    return model
