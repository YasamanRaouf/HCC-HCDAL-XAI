"""
partition_clients.py

Splits the REAL 132-sample HCC training set into 2 simulated clients, in
two ways:

  - IID split: stratified random split, each client gets roughly the same
    Dies/Lives ratio (~38%/62%). This is Week 2's default -- fewer moving
    parts while we first confirm that federating the REAL model works at
    all, before adding client heterogeneity on top.

  - Non-IID split: deliberately skewed, so one client gets a higher
    proportion of Dies patients and the other a higher proportion of Lives
    patients -- simulating real hospital heterogeneity (e.g., a referral
    center that sees sicker patients vs. a general clinic). Ready to swap
    in once the IID run is confirmed working (flip USE_NON_IID in
    run_federated_hcc.py).
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_iid(X_train, y_train, num_clients=2, seed=42):
    """Stratified split: each client gets a similar class balance."""
    if num_clients != 2:
        raise NotImplementedError("Only 2-client IID split implemented for Week 2.")

    X0, X1, y0, y1 = train_test_split(
        X_train, y_train,
        test_size=0.5,
        stratify=y_train,
        random_state=seed,
    )
    return [(X0, y0), (X1, y1)]


def split_non_iid(X_train, y_train, seed=42, client0_dies_fraction=0.75):
    """
    Deliberately skewed 2-client split: client 0 gets most of the Dies
    patients, client 1 gets most of the Lives patients (instead of a
    50/50 stratified split). The exact skew is a documented, adjustable
    parameter (client0_dies_fraction), not hidden randomness.
    """
    rng = np.random.default_rng(seed)

    dies_idx = np.where(y_train == 0)[0]
    lives_idx = np.where(y_train == 1)[0]
    rng.shuffle(dies_idx)
    rng.shuffle(lives_idx)

    n_dies_to_c0 = int(round(len(dies_idx) * client0_dies_fraction))
    n_lives_to_c0 = len(lives_idx) - int(round(len(lives_idx) * client0_dies_fraction))

    client0_idx = np.concatenate([dies_idx[:n_dies_to_c0], lives_idx[:n_lives_to_c0]])
    all_idx = set(range(len(y_train)))
    client1_idx = np.array(sorted(all_idx - set(client0_idx.tolist())))

    rng.shuffle(client0_idx)
    rng.shuffle(client1_idx)

    X0, y0 = X_train[client0_idx], y_train[client0_idx]
    X1, y1 = X_train[client1_idx], y_train[client1_idx]
    return [(X0, y0), (X1, y1)]


def print_split_summary(shards, label):
    print(f"\n--- {label} split ---")
    for i, (X, y) in enumerate(shards):
        dies = int((y == 0).sum())
        lives = int((y == 1).sum())
        print(f"Client {i}: {len(y)} samples (Dies={dies}, Lives={lives})")
