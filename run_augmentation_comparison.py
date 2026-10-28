"""
run_augmentation_comparison.py

Phase 2, Weeks 7-8. Runs the CENTRALIZED SAE-GRU-DBN pipeline (same
architecture as train.py's baseline) under four scenarios:
  1. No augmentation (baseline, reuses Phase 0's exact numbers)
  2. SMOTE
  3. SMOTE + Tomek Links (simplified BOO-ST stand-in -- see augmentation.py)
  4. CTGAN

CRITICAL: in every scenario, augmentation is fit/applied ONLY on X_train/
y_train, AFTER the train/test split already performed by
data_loader.load_data(). X_test/y_test are never touched by any
augmentation method. This is the exact rule whose violation caused
sample-level leakage in Phase 0 -- see augmentation.py's module docstring.

Usage:
    python run_augmentation_comparison.py
"""

import csv
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
)

from data_loader import load_data
from models import train_sae, train_gru, predict_gru, train_dbn, predict_dbn
from augmentation import augment_smote, augment_smote_tomek, augment_ctgan


RESULTS_CSV = "results/augmentation_comparison.csv"
CTGAN_EPOCHS = 300


def run_pipeline(X_train, y_train, X_test, y_test, scenario_name):
    """Train the SAE -> GRU/DBN -> weighted-voting pipeline once (mirrors
    train.py) and return a dict of evaluation metrics on X_test/y_test."""
    encoder = train_sae(X_train)
    X_train_sae = encoder.predict(X_train, verbose=0)
    X_test_sae = encoder.predict(X_test, verbose=0)

    gru_model = train_gru(X_train_sae, y_train)
    gru_train = predict_gru(gru_model, X_train_sae)
    gru_test = predict_gru(gru_model, X_test_sae)

    dbn_model = train_dbn(X_train_sae, y_train)
    dbn_train = predict_dbn(dbn_model, X_train_sae)
    dbn_test = predict_dbn(dbn_model, X_test_sae)

    w_gru = accuracy_score(y_train, (gru_train >= 0.5).astype(int))
    w_dbn = accuracy_score(y_train, (dbn_train >= 0.5).astype(int))
    final_proba = (w_gru * gru_test + w_dbn * dbn_test) / (w_gru + w_dbn)
    final_preds = (final_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, final_preds)
    mcc = matthews_corrcoef(y_test, final_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, final_preds, labels=[0, 1], zero_division=0
    )

    print(
        f"[{scenario_name}] acc={acc:.4f}  MCC={mcc:.4f}  "
        f"recall(Dies)={recall[0]:.4f}  recall(Lives)={recall[1]:.4f}"
    )

    return {
        "scenario": scenario_name,
        "n_train": len(y_train),
        "n_dies_train": int((y_train == 0).sum()),
        "n_lives_train": int((y_train == 1).sum()),
        "accuracy": acc,
        "mcc": mcc,
        "precision_dies": precision[0],
        "recall_dies": recall[0],
        "f1_dies": f1[0],
        "precision_lives": precision[1],
        "recall_lives": recall[1],
        "f1_lives": f1[1],
    }


def main():
    X_train, X_test, y_train, y_test, selected_features, _ = load_data()

    scenarios = []

    print("\n=== Scenario 1/4: No augmentation (baseline) ===")
    scenarios.append(run_pipeline(X_train, y_train, X_test, y_test, "none"))

    print("\n=== Scenario 2/4: SMOTE ===")
    X_sm, y_sm = augment_smote(X_train, y_train)
    scenarios.append(run_pipeline(X_sm, y_sm, X_test, y_test, "smote"))

    print("\n=== Scenario 3/4: SMOTE + Tomek (simplified BOO-ST) ===")
    X_st, y_st = augment_smote_tomek(X_train, y_train)
    scenarios.append(run_pipeline(X_st, y_st, X_test, y_test, "smote_tomek"))

    print("\n=== Scenario 4/4: CTGAN ===")
    X_ct, y_ct = augment_ctgan(X_train, y_train, selected_features, epochs=CTGAN_EPOCHS)
    scenarios.append(run_pipeline(X_ct, y_ct, X_test, y_test, "ctgan"))

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenarios[0].keys()))
        writer.writeheader()
        for row in scenarios:
            writer.writerow(row)

    print(f"\nResults written to {RESULTS_CSV}")
    print("\n=== Summary ===")
    for s in scenarios:
        print(
            f"{s['scenario']:<14} acc={s['accuracy']:.4f}  MCC={s['mcc']:.4f}  "
            f"recall(Dies)={s['recall_dies']:.4f}"
        )


if __name__ == "__main__":
    main()
