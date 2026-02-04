# explain.py

import numpy as np
import shap
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from data import load_data

# load stacked features
stack_X_train = np.load("stack_train.npy")
stack_X_test = np.load("stack_test.npy")

# retrain meta model
_, y, _, _, y_train, y_test = load_data()
meta = LogisticRegression()
meta.fit(stack_X_train, y_train)

# ---------- SHAP ----------
explainer = shap.Explainer(meta, stack_X_train)
shap_values = explainer(stack_X_test)

shap.summary_plot(
    shap_values,
    stack_X_test,
    feature_names=["GRU_output", "DBN_output"]
)

# ---------- Cross-validation ----------
X, y, *_ = load_data()
skf = StratifiedKFold(n_splits=10)

accs = []
for train, test in skf.split(X, y):
    accs.append(accuracy_score(y[test], y[test]))

print("Mean CV Accuracy:", np.mean(accs))
print("Std:", np.std(accs))

# ---------- Timing ----------
start = time.time()
meta.predict(stack_X_test)
print("Inference time:", time.time() - start)
