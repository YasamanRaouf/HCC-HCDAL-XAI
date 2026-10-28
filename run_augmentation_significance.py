"""
run_augmentation_significance.py

Phase 2, Week 9. Repeats the 4-scenario augmentation comparison
(none / SMOTE / SMOTE+Tomek / CTGAN) across multiple independent random
seeds, so results can be reported as mean +/- std rather than a single
run -- exactly the same principle already applied to the Phase 0 baseline
after the seed-instability discovery.

Each seed controls: (a) the train/test split inside data_loader.load_data,
(b) the SMOTE/SMOTE+Tomek random_state, and (c) CTGAN's random behavior
(via numpy/torch global seeding, since the `ctgan` package does not expose
its own random_state parameter in all versions -- see set_ctgan_seed()).

CTGAN_EPOCHS is reduced from the Week 7-8 default (300) to 100 here to
keep a 5-seed x 4-scenario = 20-run sweep tractable; this is a documented
runtime/thoroughness trade-off, not a silent shortcut -- increase it if
your machine has time to spare.

Usage:
    python run_augmentation_significance.py
"""

import csv
import os
import random

import numpy as np

from data_loader import load_data
from augmentation import augment_smote, augment_smote_tomek, augment_ctgan
from run_augmentation_comparison import run_pipeline


SEEDS = [0, 1, 21, 42, 100]
CTGAN_EPOCHS = 100
RESULTS_CSV = "results/augmentation_significance.csv"


def set_ctgan_seed(seed):
    """ctgan's CTGAN class does not take a random_state in every version;
    seed the underlying RNGs it draws on (numpy, python random, torch if
    present) so results are as reproducible as this library allows."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main():
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    fieldnames = None

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = None

        for seed in SEEDS:
            print(f"\n{'=' * 60}\nSEED = {seed}\n{'=' * 60}")

            X_train, X_test, y_train, y_test, selected_features, _ = load_data(random_state=seed)

            scenarios = {}

            scenarios["none"] = run_pipeline(X_train, y_train, X_test, y_test, f"none (seed={seed})")

            X_sm, y_sm = augment_smote(X_train, y_train, random_state=seed)
            scenarios["smote"] = run_pipeline(X_sm, y_sm, X_test, y_test, f"smote (seed={seed})")

            X_st, y_st = augment_smote_tomek(X_train, y_train, random_state=seed)
            scenarios["smote_tomek"] = run_pipeline(X_st, y_st, X_test, y_test, f"smote_tomek (seed={seed})")

            set_ctgan_seed(seed)
            X_ct, y_ct = augment_ctgan(X_train, y_train, selected_features, epochs=CTGAN_EPOCHS, random_state=seed)
            scenarios["ctgan"] = run_pipeline(X_ct, y_ct, X_test, y_test, f"ctgan (seed={seed})")

            for scenario_key, row in scenarios.items():
                row = dict(row)
                row["scenario"] = scenario_key  # overwrite the "(seed=...)" label used for logging
                row["seed"] = seed
                if writer is None:
                    fieldnames = ["seed"] + [k for k in row.keys() if k != "seed"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
            f.flush()

    print(f"\nAll {len(SEEDS)} seeds x 4 scenarios complete. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
