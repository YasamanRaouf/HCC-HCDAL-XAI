import numpy as np
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef

from data_loader import load_data
from models import (train_sae, train_gru, predict_gru,
                    train_dbn, predict_dbn,
                    train_meta_classifier, predict_meta)


X, y, X_train, X_test, y_train, y_test, selected_features = load_data(n_features=30)
print(f"\nData shape: train={X_train.shape}, test={X_test.shape}\n")

print("=== Training SAE ===")
encoder = train_sae(X_train, encoding_dim=32, epochs=3000)
X_train_enc = encoder.predict(X_train, verbose=0)
X_test_enc  = encoder.predict(X_test,  verbose=0)

print("\n=== Training GRU ===")
gru_model = train_gru(X_train_enc, y_train, epochs=3000)
gru_train = predict_gru(gru_model, X_train_enc)
gru_test  = predict_gru(gru_model, X_test_enc)

print("\n=== Training DBN ===")
dbn_model = train_dbn(X_train_enc, y_train)
dbn_train = predict_dbn(dbn_model, X_train_enc)
dbn_test  = predict_dbn(dbn_model, X_test_enc)

print("\n=== Training Meta-Classifier ===")
meta_model = train_meta_classifier(gru_train, dbn_train, y_train)
final_preds, final_proba = predict_meta(meta_model, gru_test, dbn_test)

print("\n" + "="*50)
print("Final Results (Meta-Classifier Ensemble):")
print("="*50)
print(f"Test Accuracy : {accuracy_score(y_test, final_preds)*100:.2f}%")
print(f"MCC           : {matthews_corrcoef(y_test, final_preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, final_preds, target_names=["Dies", "Lives"]))

stack_train = np.column_stack([gru_train, dbn_train])
stack_test  = np.column_stack([gru_test,  dbn_test])
np.save("stack_train.npy",       stack_train)
np.save("stack_test.npy",        stack_test)
np.save("y_train.npy",           y_train)
np.save("y_test.npy",            y_test)
np.save("selected_features.npy", np.array(selected_features))
