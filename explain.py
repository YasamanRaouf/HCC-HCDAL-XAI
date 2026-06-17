import numpy as np
import shap
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

stack_X_train = np.load("stack_train.npy")
stack_X_test  = np.load("stack_test.npy")
y_train       = np.load("y_train.npy")
y_test        = np.load("y_test.npy")

meta = LogisticRegression(max_iter=1000, random_state=42)
meta.fit(stack_X_train, y_train)

preds_test = meta.predict(stack_X_test)
print("Test Accuracy (Meta):", accuracy_score(y_test, preds_test))
print(classification_report(y_test, preds_test, target_names=["Dies", "Lives"]))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []
for tr_idx, te_idx in skf.split(stack_X_train, y_train):
    meta_cv = LogisticRegression(max_iter=1000, random_state=42)
    meta_cv.fit(stack_X_train[tr_idx], y_train[tr_idx])
    preds_cv = meta_cv.predict(stack_X_train[te_idx])
    cv_scores.append(accuracy_score(y_train[te_idx], preds_cv))

print(f"\nCV Accuracy (10-Fold): {np.mean(cv_scores)*100:.2f}% +/- {np.std(cv_scores)*100:.2f}%")

explainer   = shap.LinearExplainer(meta, stack_X_train)
shap_values = explainer.shap_values(stack_X_test)

shap.summary_plot(shap_values, stack_X_test,
                  feature_names=["GRU_output", "DBN_output"], show=False)
plt.savefig("shap_summary.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: shap_summary.png")

start = time.perf_counter()
_ = meta.predict(stack_X_test)
elapsed = time.perf_counter() - start
print(f"Inference time: {elapsed*1000:.4f} ms")
