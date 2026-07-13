# train.py

import os

# Must be set before importing tensorflow/keras to be fully effective.
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef

# tf.keras.utils.set_random_seed seeds Python's `random`, NumPy, and
# TensorFlow/Keras all at once. Without this, SAE/GRU weight initialization
# and mini-batch shuffling are non-deterministic, so the SAME code on the
# SAME data can give a different accuracy/MCC on every run (as observed:
# 75.76% vs 69.70% across two runs with identical data_loader.py output).
import tensorflow as tf
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# set_random_seed() alone is not always enough: on some backends (notably
# Apple Silicon's Metal GPU plugin) individual ops can still execute in a
# non-deterministic order due to parallelism, which can flip a handful of
# borderline predictions between runs even with identical seeds/data. This
# forces TensorFlow to use deterministic op implementations wherever
# available.
tf.config.experimental.enable_op_determinism()

from data_loader import load_data
from models import train_sae, train_gru, predict_gru, train_dbn, predict_dbn


# NOTE: load_data() now loads the RAW hcc-data.csv (165 patients, 63/102
# imbalance) and returns the fitted preprocessing pipeline as well, since all
# imputation / scaling / feature-selection is fit only on the training fold.
X_train, X_test, y_train, y_test, selected_features, preprocessing_pipeline = load_data()

# SAE encoding
encoder = train_sae(X_train)
X_train_sae = encoder.predict(X_train, verbose=0)
X_test_sae  = encoder.predict(X_test,  verbose=0)

# GRU
gru_model = train_gru(X_train_sae, y_train)
gru_train  = predict_gru(gru_model, X_train_sae)
gru_test   = predict_gru(gru_model, X_test_sae)

# DBN
dbn_model = train_dbn(X_train_sae, y_train)
dbn_train  = predict_dbn(dbn_model, X_train_sae)
dbn_test   = predict_dbn(dbn_model, X_test_sae)

# Weighted Voting (weight = accuracy on train set)
w_gru = accuracy_score(y_train, (gru_train >= 0.5).astype(int))
w_dbn = accuracy_score(y_train, (dbn_train >= 0.5).astype(int))

final_proba = (w_gru * gru_test + w_dbn * dbn_test) / (w_gru + w_dbn)
final_preds = (final_proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, final_preds))
print("MCC:",      matthews_corrcoef(y_test, final_preds))
print(classification_report(y_test, final_preds))

# save for explain.py
stack_X_train = np.column_stack([gru_train, dbn_train])
stack_X_test  = np.column_stack([gru_test,  dbn_test])
np.save("stack_train.npy", stack_X_train)
np.save("stack_test.npy",  stack_X_test)
np.save("y_train.npy",     y_train)
np.save("y_test.npy",      y_test)
np.save("weights.npy",     np.array([w_gru, w_dbn]))
