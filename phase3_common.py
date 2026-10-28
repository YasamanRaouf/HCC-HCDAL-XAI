"""
phase3_common.py

Shared building blocks for the full Phase 3 run (Design B, approved, with
the Stability amendment). Everything here is lifted UNCHANGED from
pilot_phase3.py, which was already executed successfully and validated
(wrapper == standard pipeline, SHAP additivity ~1e-16, no errors). This
module exists so run_phase3_scenario.py (25 folds) and the pilot never
drift apart -- one implementation, reused by both.

Nothing in this file trains a model, runs a fold, or calls CTGAN by
itself; it only defines functions/classes that the run script calls.
"""

import time
from functools import partial

import numpy as np
import tensorflow as tf
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from models import predict_dbn


SEED = 42
N_FEATURES = 30
N_BACKGROUND = 20        # shap.kmeans(X_train, 20) -- approved Design B value
N_PERMUTATIONS = 10      # permutations per explained sample -- approved Design B value
PFI_N_REPEATS = 10       # repeats for sklearn permutation_importance (Consistency)


def build_fold_preprocessor(seed=SEED, n_features=N_FEATURES):
    """
    Exactly the same three steps as data_loader.load_data()'s Pipeline
    (SimpleImputer(mean) -> MinMaxScaler -> SelectKBest(seeded mutual_info)),
    but returned unfitted so the caller fits it on a CV fold's train part
    only (data_loader.load_data() itself is not reused because it hard-codes
    a single 80/20 split, not a CV fold).
    """
    seeded_mi = partial(mutual_info_classif, random_state=seed)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("selector", SelectKBest(seeded_mi, k=n_features)),
    ])


class FinalModel:
    """
    The production ensemble as ONE function over the preprocessed clinical
    features:  X -> SAE encoder -> {GRU proba, DBN proba} -> weighted vote.
    Identical to the class validated in pilot_phase3.py (wrapper output was
    bit-for-bit equal to the standard models.py prediction path, max abs
    diff = 0.0). Keras models are called directly (not .predict()) to avoid
    per-call tf.function overhead when SHAP invokes this thousands of times.
    """

    def __init__(self, encoder, gru_model, dbn_model, w_gru, w_dbn):
        self.encoder = encoder
        self.gru = gru_model
        self.dbn = dbn_model
        self.w_gru = float(w_gru)
        self.w_dbn = float(w_dbn)
        self.n_calls = 0
        self.n_rows = 0

    def reset_counters(self):
        self.n_calls = 0
        self.n_rows = 0

    def predict_final_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.n_calls += 1
        self.n_rows += int(X.shape[0])

        z = self.encoder(tf.convert_to_tensor(X), training=False).numpy()
        z_seq = z.reshape(z.shape[0], 1, z.shape[1])
        p_gru = self.gru(tf.convert_to_tensor(z_seq), training=False).numpy().reshape(-1)
        p_dbn = predict_dbn(self.dbn, z)
        return (self.w_gru * p_gru + self.w_dbn * p_dbn) / (self.w_gru + self.w_dbn)

    def predict_final_class(self, X):
        return (self.predict_final_proba(X) >= 0.5).astype(int)


def check_wrapper_matches_reference(model, X_val, ref_final_proba, tol=1e-4):
    """Same equivalence check as the pilot: refuse to explain a function
    that does not match the standard evaluation path. Returns (ok, max_diff)
    instead of raising, so the caller can log-and-skip a bad fold instead of
    crashing a 25-fold run over one fold's numerical issue."""
    wrapped = model.predict_final_proba(X_val)
    max_diff = float(np.max(np.abs(ref_final_proba - wrapped)))
    return max_diff < tol, max_diff


def make_permutation_explainer(model, X_train, selected_features,
                                n_background=N_BACKGROUND,
                                n_permutations=N_PERMUTATIONS, seed=SEED):
    """Exactly the explainer construction validated in the pilot:
    shap.PermutationExplainer with an Independent masker over
    shap.kmeans(X_train, n_background) centroids, and max_evals set so
    every sample gets exactly n_permutations permutations."""
    n_feat = X_train.shape[1]
    max_evals = n_permutations * (2 * n_feat + 1)
    background = shap.kmeans(X_train, n_background).data
    masker = shap.maskers.Independent(background, max_samples=n_background)
    try:
        explainer = shap.PermutationExplainer(
            model.predict_final_proba, masker, feature_names=selected_features, seed=seed)
    except TypeError:  # older SHAP without the `seed` kwarg
        explainer = shap.PermutationExplainer(
            model.predict_final_proba, masker, feature_names=selected_features)
    return explainer, max_evals


def explain_fold(explainer, X_val, max_evals):
    """Run the (already-built) explainer on one fold's out-of-fold samples
    and time it. Returns (shap_values [n_val, n_features], base_values,
    seconds)."""
    t0 = time.perf_counter()
    explanation = explainer(X_val, max_evals=max_evals, silent=True)
    seconds = time.perf_counter() - t0
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values).reshape(-1)
    return values, base, seconds


def permutation_feature_importance(model, X_val, y_val, n_repeats=PFI_N_REPEATS, seed=SEED):
    """
    Consistency reference (approved design: Consistency = agreement between
    SHAP and the pipeline's OWN sensitivity to each feature, via
    Permutation Feature Importance on predict_final_proba -- NOT a
    comparison against an independent Random Forest; see Phase 3 audit,
    section 3/CONSISTENCY REASSESSMENT).

    Scored on negative log-loss (lower loss = better; sklearn's
    permutation_importance negates the scorer internally when
    `scoring` decreases-is-better, so we pass a callable explicitly).
    """
    def neg_log_loss_scorer(estimator, X, y):
        proba = estimator.predict_final_proba(X)
        proba = np.clip(proba, 1e-7, 1 - 1e-7)
        return -log_loss(y, proba, labels=[0, 1])

    # permutation_importance expects an object with .fit/.predict OR accepts
    # any estimator-like object when passed a custom scorer that only calls
    # the methods we actually implement; FinalModel has no .fit (by design,
    # it wraps an already-trained pipeline), so we adapt with a thin shim.
    class _Adapter:
        def __init__(self, m):
            self._m = m

        def predict_final_proba(self, X):
            return self._m.predict_final_proba(X)

        def fit(self, *a, **k):
            return self  # no-op: already trained

    result = permutation_importance(
        _Adapter(model), X_val, y_val,
        scoring=neg_log_loss_scorer,
        n_repeats=n_repeats,
        random_state=seed,
    )
    return result.importances_mean  # one value per feature, same order as X_val columns
