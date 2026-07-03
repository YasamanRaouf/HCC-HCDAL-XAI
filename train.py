import numpy as np
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
from data_loader import load_data
from models import train_sae, train_gru, predict_gru, train_dbn, predict_dbn, train_meta_classifier, predict_meta


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


set_seed(42)

X_train, X_test, y_train, y_test, selected_features, pipeline = load_data(
    path="Data/hcc-data-complete-balanced.csv",
    n_features=30,
    random_state=42
)

print("\n[SAE Training]")
encoder = train_sae(X_train, encoding_dim=32, epochs=3000)
X_train_enc = encoder.predict(X_train)
X_test_enc = encoder.predict(X_test)
print(f"Encoded shape: {X_train_enc.shape}")

print("\n[GRU Training]")
gru_model = train_gru(X_train_enc, y_train, epochs=3000)
gru_train = predict_gru(gru_model, X_train_enc)
gru_test = predict_gru(gru_model, X_test_enc)

print("\n[DBN Training]")
dbn_model = train_dbn(X_train_enc, y_train)
dbn_train = predict_dbn(dbn_model, X_train_enc)
dbn_test = predict_dbn(dbn_model, X_test_enc)

print("\n[Meta-Classifier Training]")
meta_model = train_meta_classifier(gru_train, dbn_train, y_train)

final_preds, final_proba = predict_meta(meta_model, gru_test, dbn_test)

print("\n" + "="*60)
print("Final Test Results")
print("="*60)
acc = accuracy_score(y_test, final_preds)
mcc = matthews_corrcoef(y_test, final_preds)
print(f"Accuracy: {acc:.4f}")
print(f"MCC: {mcc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, final_preds, target_names=["Dies", "Lives"]))

stack_train = np.column_stack([gru_train, dbn_train])
stack_test = np.column_stack([gru_test, dbn_test])

np.save("stack_train.npy", stack_train)
np.save("stack_test.npy", stack_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("selected_features.npy", selected_features)

print("\nStacked outputs saved.")
