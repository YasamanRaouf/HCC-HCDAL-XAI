# models.py

import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras import regularizers
from pydbn import DBN


# ---------- SAE ----------
def train_sae(X_train, encoding_dim=32):

    input_dim = X_train.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(
        encoding_dim,
        activation='relu',
        activity_regularizer=regularizers.l1(1e-4)
    )(input_layer)

    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    sae = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    sae.compile(optimizer='adam', loss='mse')
    sae.fit(X_train, X_train, epochs=100, batch_size=16, verbose=1)

    return encoder


# ---------- GRU ----------
def train_gru(X_train, y_train):

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

    model = Sequential([
        GRU(64, input_shape=(1, X_train.shape[2])),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

    return model


def predict_gru(model, X):
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return model.predict(X).flatten()


# ---------- DBN ----------
def train_dbn(X_train, y_train):

    dbn = DBN(
        hidden_layers_structure=[64, 32],
        learning_rate_rbm=0.01,
        learning_rate=0.01,
        n_epochs_rbm=50,
        n_iter_backprop=100,
        batch_size=16,
        activation_function='relu'
    )

    dbn.fit(X_train, y_train)
    return dbn


def predict_dbn(model, X):
    return model.predict_proba(X)[:, 1]
