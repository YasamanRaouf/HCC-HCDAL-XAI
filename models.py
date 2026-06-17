import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_sae(X_train, encoding_dim=32, epochs=3000):
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(1e-4))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    sae = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    sae.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    sae.fit(X_train, X_train,
            epochs=epochs,
            batch_size=5,
            validation_split=0.1,
            callbacks=[es],
            verbose=0)
    print("[SAE] Training complete")
    return encoder


def train_gru(X_train, y_train, epochs=3000):
    X_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    model = Sequential([
        GRU(64, input_shape=(1, X_train.shape[1])),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)
    model.fit(X_3d, y_train,
              epochs=epochs,
              batch_size=5,
              validation_split=0.1,
              callbacks=[es],
              verbose=0)
    print("[GRU] Training complete")
    return model


def predict_gru(model, X):
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    return model.predict(X_3d, verbose=0).flatten()


def train_dbn(X_train, y_train):
    rbm1 = BernoulliRBM(n_components=64, learning_rate=0.01,
                        n_iter=100, batch_size=5, random_state=42)
    rbm2 = BernoulliRBM(n_components=32, learning_rate=0.01,
                        n_iter=100, batch_size=5, random_state=42)
    clf  = LogisticRegression(max_iter=1000, random_state=42)
    dbn  = Pipeline([('rbm1', rbm1), ('rbm2', rbm2), ('clf', clf)])
    dbn.fit(X_train, y_train)
    print("[DBN] Training complete")
    return dbn


def predict_dbn(model, X):
    return model.predict_proba(X)[:, 1]


def train_meta_classifier(gru_proba_train, dbn_proba_train, y_train):
    stack_train = np.column_stack([gru_proba_train, dbn_proba_train])
    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(stack_train, y_train)
    print("[Meta-Classifier] Training complete")
    return meta


def predict_meta(meta_model, gru_proba, dbn_proba):
    stack = np.column_stack([gru_proba, dbn_proba])
    return meta_model.predict(stack), meta_model.predict_proba(stack)[:, 1]
