# explain.py

import numpy as np
import shap
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

stack_X_train = np.load("stack_train.npy")
stack_X_test  = np.load("stack_test.npy")
y_train       = np.load("y_train.npy")
y_test        = np.load("y_test.npy")
weights       = np.load("weights.npy")
w_gru, w_dbn  = weights[0], weights[1]

# --- Cross-Validation با Weighted Voting (مطابق معماری مقاله) ---
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []
for _, te in skf.split(stack_X_train, y_train):
    g, d = stack_X_train[te, 0], stack_X_train[te, 1]
    preds = ((w_gru * g + w_dbn * d) / (w_gru + w_dbn) >= 0.5).astype(int)
    cv_scores.append(accuracy_score(y_train[te], preds))

print(f"CV Accuracy (Weighted Voting): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# --- SHAP با LogisticRegression به عنوان surrogate ---
surrogate = LogisticRegression(max_iter=200).fit(stack_X_train, y_train)
explainer   = shap.LinearExplainer(surrogate, stack_X_train)
shap_values = explainer.shap_values(stack_X_test)

shap.summary_plot(shap_values, stack_X_test,
                  feature_names=["GRU_output", "DBN_output"], show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
print("Saved: shap_summary.png")

# --- Inference time ---
start = time.time()
surrogate.predict(stack_X_test)
print(f"Inference time: {time.time() - start:.6f}s")
