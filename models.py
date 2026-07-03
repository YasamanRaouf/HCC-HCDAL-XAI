import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
import tensorflow as tf



def train_sae(X_train, encoding_dim=32, epochs=3000):
    input_dim = X_train.shape[1]
    
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=l1(1e-4))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    
    autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=16,
        shuffle=True,
        verbose=0,
        callbacks=[early_stop]
    )
    
    print(f"SAE trained: input_dim={input_dim}, encoding_dim={encoding_dim}")
    return encoder


def train_gru(X_train, y_train, epochs=3000):
    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    model = tf.keras.Sequential([
        GRU(64, input_shape=(1, X_train.shape[1])),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    
    model.fit(
        X_train_reshaped, y_train,
        epochs=epochs,
        batch_size=16,
        verbose=0,
        callbacks=[early_stop]
    )
    
    print(f"GRU trained: {model.count_params()} parameters")
    return model


def predict_gru(model, X):
    X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
    return model.predict(X_reshaped, verbose=0).flatten()


def train_dbn(X_train, y_train):
    rbm = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=50, random_state=42)
    logistic = LogisticRegression(max_iter=100, solver='lbfgs', random_state=42)
    
    classifier = SkPipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    classifier.fit(X_train, y_train)
    
    print(f"DBN trained: RBM components={rbm.n_components}")
    return classifier


def predict_dbn(model, X):
    return model.predict_proba(X)[:, 1]


def train_meta_classifier(gru_output, dbn_output, y_train):
    stacked = np.column_stack([gru_output, dbn_output])
    meta = LogisticRegression(max_iter=500, random_state=42)
    meta.fit(stacked, y_train)
    print("Meta-classifier trained")
    return meta


def predict_meta(meta_model, gru_output, dbn_output):
    stacked = np.column_stack([gru_output, dbn_output])
    preds = meta_model.predict(stacked)
    proba = meta_model.predict_proba(stacked)[:, 1]
    return preds, proba
