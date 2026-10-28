#explain.py
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


stack_train = np.load("stack_train.npy")
stack_test = np.load("stack_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

meta_model = LogisticRegression(max_iter=500, random_state=42)
meta_model.fit(stack_train, y_train)

explainer = shap.LinearExplainer(meta_model, stack_train)
shap_values = explainer.shap_values(stack_test)

feature_names = ["GRU_output", "DBN_output"]

shap.summary_plot(shap_values, stack_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()

shap.summary_plot(shap_values, stack_test, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=300, bbox_inches='tight')
plt.close()

print("SHAP plots saved: shap_summary.png, shap_importance.png")
