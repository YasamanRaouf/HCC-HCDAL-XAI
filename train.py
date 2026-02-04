# train.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef

from data import load_data
from models import train_sae, train_gru, predict_gru, train_dbn, predict_dbn


# load
X, y, X_train, X_test, y_train, y_test = load_data()

# SAE
encoder = train_sae(X_train)
X_train_sae = encoder.predict(X_train)
X_test_sae = encoder.predict(X_test)

# GRU
gru_model = train_gru(X_train_sae, y_train)
gru_train = predict_gru(gru_model, X_train_sae)
gru_test = predict_gru(gru_model, X_test_sae)

# DBN
dbn = train_dbn(X_train_sae, y_train)
dbn_train = predict_dbn(dbn, X_train_sae)
dbn_test = predict_dbn(dbn, X_test_sae)

# stacking
stack_X_train = np.column_stack([gru_train, dbn_train])
stack_X_test = np.column_stack([gru_test, dbn_test])

meta = LogisticRegression()
meta.fit(stack_X_train, y_train)

final_preds = meta.predict(stack_X_test)

# evaluation
print("Accuracy:", accuracy_score(y_test, final_preds))
print("MCC:", matthews_corrcoef(y_test, final_preds))
print(classification_report(y_test, final_preds))

# save for explain.py
np.save("stack_train.npy", stack_X_train)
np.save("stack_test.npy", stack_X_test)
np.save("final_preds.npy", final_preds)
