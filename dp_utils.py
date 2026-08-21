"""
dp_utils.py

Lightweight differential-privacy MECHANISM for the federated SAE+GRU
pipeline: per-client update clipping + calibrated Gaussian noise, applied
to each client's local parameter update before it is sent to the server
for aggregation (a standard "DP-FedAvg" style approach, McMahan et al.
2017-ish).

IMPORTANT SCOPE NOTE (read before reporting any epsilon number in the
thesis): this implements the DP *mechanism* -- clip + noise -- which is
the necessary building block. It does NOT yet implement formal privacy
ACCOUNTING (turning noise_multiplier / clip_norm / num_rounds /
num_clients into a rigorous (epsilon, delta) guarantee via e.g. a moments
accountant or Renyi-DP accountant). Reporting a bare "epsilon = X" number
without a real accountant behind it would be a methodological overclaim.
For now, results should be reported and compared in terms of the
mechanism's own parameters (clip_norm, noise_multiplier), with formal
(epsilon, delta) accounting flagged explicitly as follow-up work.
"""

import numpy as np


def compute_update(old_params, new_params):
    """Per-array delta: new - old, one array per model weight tensor."""
    return [new - old for old, new in zip(old_params, new_params)]


def global_l2_norm(arrays):
    """L2 norm of the update treated as one flattened vector across all
    weight tensors (not per-layer) -- this is what clip_norm bounds."""
    total = 0.0
    for a in arrays:
        total += float(np.sum(np.square(a)))
    return float(np.sqrt(total))


def clip_update(update, clip_norm: float):
    """Scale the whole update down (uniformly, across all arrays) so its
    global L2 norm does not exceed clip_norm. This bounds the sensitivity
    of a single client's contribution -- the prerequisite for calibrating
    DP noise. Returns (clipped_update, pre_clip_norm) so callers can log
    how often/how hard clipping actually engaged."""
    norm = global_l2_norm(update)
    if norm <= clip_norm or norm == 0.0:
        return update, norm
    scale = clip_norm / norm
    return [u * scale for u in update], norm


def add_gaussian_noise(update, clip_norm: float, noise_multiplier: float, rng: np.random.Generator):
    """Add i.i.d. Gaussian noise, std = noise_multiplier * clip_norm, to
    every element of the (already-clipped) update. Larger noise_multiplier
    -> stronger privacy protection, larger accuracy cost."""
    std = noise_multiplier * clip_norm
    noised = []
    for u in update:
        noise = rng.normal(loc=0.0, scale=std, size=u.shape).astype(u.dtype)
        noised.append(u + noise)
    return noised


def apply_dp(old_params, new_params, clip_norm: float, noise_multiplier: float, seed: int):
    """
    Full client-side DP step for one round:
      1. compute this round's local update (new_params - old_params)
      2. clip its global L2 norm to `clip_norm` (bounds sensitivity)
      3. add calibrated Gaussian noise (std = noise_multiplier * clip_norm)
      4. return old_params + noised_clipped_update -- the "DP-protected"
         parameters that get sent to the server instead of the raw ones.

    If noise_multiplier == 0.0, this reduces to plain clipping with no
    noise (a useful clip-only control condition for comparison).
    Returns (protected_params, pre_clip_norm) -- log pre_clip_norm across
    rounds/clients to see how often clipping actually engages.
    """
    rng = np.random.default_rng(seed)
    update = compute_update(old_params, new_params)
    clipped_update, raw_norm = clip_update(update, clip_norm)
    if noise_multiplier > 0.0:
        protected_update = add_gaussian_noise(clipped_update, clip_norm, noise_multiplier, rng)
    else:
        protected_update = clipped_update
    protected_params = [old + u for old, u in zip(old_params, protected_update)]
    return protected_params, raw_norm


def total_bytes(arrays):
    """Total size in bytes of a list of NumPy arrays -- used for the
    communication-overhead metric (bytes exchanged per round)."""
    return int(sum(a.nbytes for a in arrays))
