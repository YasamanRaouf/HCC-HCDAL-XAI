"""
augmentation.py

Fold-safe data augmentation for the minority ("Dies") class. ALL THREE
functions below must be called ONLY with a training fold (X_train, y_train)
-- NEVER with any data that includes the held-out test set. This is the
exact rule whose violation (in the pre-balanced dataset artifact discovered
in Phase 0) caused sample-level leakage; it is repeated here deliberately.

Three methods are compared in Phase 2:
  1. augment_smote      -- classic SMOTE (Chawla et al., 2002)
  2. augment_smote_tomek -- SMOTE + Tomek Links cleanup (a documented,
     SIMPLIFIED stand-in for the BOO-ST method of paper [8]; see the scope
     note below)
  3. augment_ctgan      -- a modern conditional tabular GAN (Xu et al., 2019)

SCOPE NOTE on augment_smote_tomek vs. full BOO-ST: paper [8]'s BOO-ST is a
three-stage method (Boosting-based minority reweighting -> SMOTE -> Tomek
Links). The boosting-based reweighting stage (which adjusts sample
importance before SMOTE using an AdaBoost-style procedure) is NOT
implemented here; augment_smote_tomek implements only the SMOTE + Tomek
Links stages. This is an intentional, documented scope reduction, not a
silent approximation -- it should be described as such in the thesis
(e.g., "SMOTE+Tomek, a simplified two-stage variant of BOO-ST") rather than
claimed as a full BOO-ST reproduction.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from ctgan import CTGAN


def augment_smote(X_train, y_train, random_state=42):
    """Classic SMOTE oversampling of the minority class up to parity."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


def augment_smote_tomek(X_train, y_train, random_state=42):
    """SMOTE followed by Tomek Links cleanup (simplified BOO-ST stand-in --
    see module docstring for the scope note on what is NOT implemented)."""
    smt = SMOTETomek(random_state=random_state)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    return X_res, y_res


def augment_ctgan(X_train, y_train, feature_names, target_col="Class",
                   epochs=300, random_state=42):
    """
    Fit a CTGAN model ONLY on the training fold, then generate synthetic
    minority ("Dies", label 0) samples until the class is balanced.

    X_train is expected to already be preprocessed (imputed, scaled to
    [0,1], 30 selected features) -- i.e., the output of the SAME
    data_loader.load_data() pipeline used everywhere else in this project,
    so synthetic samples live in the same feature space as the real ones.
    """
    df = pd.DataFrame(X_train, columns=feature_names)
    df[target_col] = y_train

    n_majority = int((y_train == 1).sum())
    n_minority = int((y_train == 0).sum())
    n_to_generate = n_majority - n_minority
    if n_to_generate <= 0:
        return X_train, y_train  # already balanced or majority is the minority; nothing to do

    ctgan = CTGAN(epochs=epochs, verbose=False)
    # Only discrete_columns needs specifying for CTGAN; treat the target as
    # the sole categorical column since the 30 selected features are all
    # continuous after preprocessing.
    ctgan.fit(df, discrete_columns=[target_col])

    # Sample extra rows and keep only the ones CTGAN itself labeled as the
    # minority class (label 0 / "Dies"), then trim to exactly n_to_generate.
    synthetic = ctgan.sample(n_to_generate * 3)
    synthetic_minority = synthetic[synthetic[target_col] == 0]
    if len(synthetic_minority) < n_to_generate:
        # CTGAN under-produced minority-labeled rows; sample more.
        extra = ctgan.sample(n_to_generate * 5)
        synthetic_minority = pd.concat([
            synthetic_minority, extra[extra[target_col] == 0]
        ])
    synthetic_minority = synthetic_minority.iloc[:n_to_generate]

    X_synth = synthetic_minority[feature_names].values
    y_synth = synthetic_minority[target_col].values.astype(int)

    X_res = np.vstack([X_train, X_synth])
    y_res = np.concatenate([y_train, y_synth])
    return X_res, y_res


def check_synthetic_quality(X_real_minority, X_synth_minority, feature_names, top_n=5):
    """
    Quick sanity check (Week 7 deliverable): compare per-feature mean/std of
    real vs. synthetic minority-class samples. Large discrepancies indicate
    low-quality or mode-collapsed synthetic data -- a known risk for GANs on
    tiny tabular datasets (~50 real minority samples here), and something to
    look for explicitly rather than assume away.
    """
    real_mean = X_real_minority.mean(axis=0)
    synth_mean = X_synth_minority.mean(axis=0)
    real_std = X_real_minority.std(axis=0)
    synth_std = X_synth_minority.std(axis=0)

    mean_diff = np.abs(real_mean - synth_mean)
    worst_idx = np.argsort(-mean_diff)[:top_n]

    print(f"{'Feature':<15} {'Real mean':>10} {'Synth mean':>11} {'Real std':>9} {'Synth std':>10}")
    for i in worst_idx:
        print(f"{feature_names[i]:<15} {real_mean[i]:>10.3f} {synth_mean[i]:>11.3f} "
              f"{real_std[i]:>9.3f} {synth_std[i]:>10.3f}")
    return worst_idx
