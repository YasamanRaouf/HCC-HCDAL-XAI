"""
pilot_phase3.py

Phase 3 -- ONE-FOLD FEASIBILITY PILOT (Design B, approved with the Stability
amendment). This is NOT the Phase 3 analysis. It exists only to measure, on a
single fold, whether Design B is affordable before committing to all 25 folds:

  1. Train the full production pipeline (SAE -> GRU + DBN -> weighted voting)
     on the TRAIN part of ONE fold of the planned Repeated Stratified 5-Fold
     x 5 CV (split #PILOT_SPLIT_INDEX, default 0).
  2. Wrap the trained pipeline as a single function predict_final_proba(X)
     over the 30 preprocessed clinical features, and check that it is
     numerically identical to the standard models.py prediction path.
  3. Run shap.PermutationExplainer (background = shap.kmeans(X_train, 20))
     on the OUT-OF-FOLD samples of the same fold.
  4. Record: training time per component, SHAP time, number of model
     evaluations, memory usage, every warning, every error.

NOT run here (on purpose): the other 24 folds, the None-vs-CTGAN comparison,
statistical tests, Generalizability/Consistency/Stability computation.

Existing project code is reused unchanged: COLUMN_NAMES from data_loader.py,
train_sae / train_gru / predict_gru / train_dbn / predict_dbn from models.py.
data_loader.load_data() itself is NOT called, because it hard-codes a single
80/20 split; the preprocessing Pipeline below reproduces its steps exactly
(SimpleImputer(mean) -> MinMaxScaler -> SelectKBest(seeded mutual_info, k=30)),
fit on the fold's TRAIN part only.

HOW TO RUN (from the project folder, in the hcc_thesis environment):
    conda activate hcc_thesis
    cd /Users/yasaman/master_thesis_HCC/HepatocellularCarcinomaDet
    python pilot_phase3.py

Optional environment variables:
    PILOT_SPLIT_INDEX   which of the 25 CV splits to use (default 0)
    PILOT_MAX_SHAP_MIN  abort the full-fold SHAP run if the warm-up projects
                        more than this many minutes for the fold (default 45)
    HCC_DATA_PATH       path to the raw hcc-data.csv (default Data/hcc-data.csv)

Output: a console report + results/pilot_phase3_report.json
"""

import os

# Must be set before numpy/tensorflow are imported (same as train.py).
os.environ.setdefault("PYTHONHASHSEED", "0")

import sys
import time
import json
import platform
import traceback
import warnings
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
_determinism_error = None
try:
    tf.config.experimental.enable_op_determinism()
except Exception as exc:  # recorded in the report instead of crashing the pilot
    _determinism_error = repr(exc)

import shap
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score

from data_loader import COLUMN_NAMES
from models import train_sae, train_gru, predict_gru, train_dbn, predict_dbn


# ----------------------------------------------------------------------------
# Pilot configuration (mirrors the approved Design B)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT_SPLIT_INDEX = int(os.environ.get("PILOT_SPLIT_INDEX", "0"))
ABORT_IF_FOLD_SHAP_EXCEEDS_MIN = float(os.environ.get("PILOT_MAX_SHAP_MIN", "45"))
DATA_PATH = os.environ.get("HCC_DATA_PATH", "Data/hcc-data.csv")

N_SPLITS = 5
N_REPEATS = 5
N_FEATURES = 30
N_BACKGROUND = 20          # shap.kmeans(X_train, 20)
N_PERMUTATIONS = 10        # permutations per explained sample
WARMUP_SAMPLES = 2         # used only to project the cost of the full fold
CLASS_LIVES, CLASS_DIES = 1, 0

REPORT_PATH = os.path.join(HERE, "results", "pilot_phase3_report.json")


# ----------------------------------------------------------------------------
# Small measurement helpers
# ----------------------------------------------------------------------------
def rss_mb():
    """Current resident memory of this process in MB (None if unavailable)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return None


def peak_rss_mb():
    """Peak resident memory of this process in MB (None if unavailable)."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        return value / 1e6 if sys.platform == "darwin" else value / 1e3
    except Exception:
        return None


@contextmanager
def record_warnings(store, stage):
    """Capture every warning raised inside the block, grouped by message."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for w in caught:
                key = "[%s] %s: %s" % (stage, w.category.__name__, str(w.message)[:160])
                store[key] += 1


@contextmanager
def timed(report, name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        report["timings_sec"][name] = time.perf_counter() - t0


# ----------------------------------------------------------------------------
# The single callable that SHAP explains
# ----------------------------------------------------------------------------
class FinalModel:
    """
    The production ensemble as ONE function over the 30 preprocessed clinical
    features:

        X -> SAE encoder -> { GRU proba, DBN proba }
          -> (w_gru * p_gru + w_dbn * p_dbn) / (w_gru + w_dbn)

    where w_gru / w_dbn are the train-set accuracies of the two sub-models
    (identical to the weighting used in train.py / run_pipeline). Keras models
    are called directly (not .predict) to avoid the per-call tf.function
    overhead/retracing when SHAP invokes this function thousands of times.
    Every call is counted so the number of model evaluations can be reported.
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


# ----------------------------------------------------------------------------
# Console summary
# ----------------------------------------------------------------------------
def fmt_min(seconds):
    return "%.1f min (%.2f h)" % (seconds / 60.0, seconds / 3600.0)


def print_summary(report):
    line = "=" * 72
    print("\n" + line)
    print("PHASE 3 PILOT -- FEASIBILITY REPORT")
    print(line)

    env = report.get("environment", {})
    print("\n[Environment]")
    for key in ["python", "conda_env", "platform", "tensorflow", "keras",
                "sklearn", "shap", "numpy", "pandas", "tf_devices"]:
        print("  %-12s %s" % (key, env.get(key)))
    for note in env.get("notes", []):
        print("  !! " + note)

    fold = report.get("fold", {})
    if fold:
        print("\n[Fold] split #%s of %s (Repeated Stratified %d-Fold x %d)" % (
            fold.get("split_index"), fold.get("n_total_splits"), N_SPLITS, N_REPEATS))
        print("  train: %s samples (Dies=%s)   out-of-fold: %s samples (Dies=%s)" % (
            fold.get("n_train"), fold.get("n_dies_train"),
            fold.get("n_val"), fold.get("n_dies_val")))
        print("  selected features: %s" % fold.get("n_selected_features"))

    t = report.get("timings_sec", {})
    if t:
        print("\n[Timings, seconds]")
        for key, value in t.items():
            print("  %-28s %10.2f" % (key, value))

    wc = report.get("wrapper_check")
    if wc:
        print("\n[Wrapper == standard pipeline?]")
        print("  max |wrapper - models.py path| on out-of-fold = %.2e  -> %s" % (
            wc["max_abs_diff"], "OK" if wc["ok"] else "MISMATCH"))
        print("  fold sanity (not a result): acc=%.3f  MCC=%.3f  recall(Dies)=%.3f" % (
            wc["fold_accuracy"], wc["fold_mcc"], wc["fold_recall_dies"]))

    shap_info = report.get("shap", {})
    if shap_info:
        print("\n[SHAP: PermutationExplainer, background=kmeans(%d), %d permutations/sample]" % (
            N_BACKGROUND, N_PERMUTATIONS))
        warm = shap_info.get("warmup")
        if warm:
            print("  warm-up (%d samples): %.2f s  ->  %.2f s/sample, %d model calls, %d model rows" % (
                warm["n_samples"], warm["seconds"], warm["sec_per_sample"],
                warm["model_calls"], warm["model_rows"]))
            print("  projected full-fold SHAP: %s" % fmt_min(warm["projected_fold_seconds"]))
        full = shap_info.get("full_fold")
        if full:
            print("  FULL FOLD (%d samples): %.2f s  ->  %.2f s/sample" % (
                full["n_samples"], full["seconds"], full["sec_per_sample"]))
            print("    model calls (batched): %d   model rows scored: %d  (%.0f rows/sample)" % (
                full["model_calls"], full["model_rows"],
                full["model_rows"] / max(full["n_samples"], 1)))
            print("    theoretical masked evaluations: %d" % full["theoretical_masked_evals"])
            print("    additivity check max |sum(SHAP)+base - f(x)| = %.2e" % full["additivity_max_abs_err"])
            print("    top-5 features by mean |SHAP| (sanity only): %s" % ", ".join(full["top5_features"]))
        elif shap_info.get("full_fold_skipped"):
            print("  FULL FOLD SKIPPED: %s" % shap_info["full_fold_skipped"])

    proj = report.get("projection")
    if proj:
        print("\n[Projection for Design B from this one fold]")
        print("  one fold (train + SHAP)                 : %s" % fmt_min(proj["per_fold_seconds"]))
        print("  25 folds, scenario 'none' only          : %s" % fmt_min(proj["none_25_folds_seconds"]))
        print("  25 folds x 2 scenarios (none + ctgan)   : %s" % fmt_min(proj["two_scenarios_seconds"]))
        print("  NOTE: CTGAN training time is NOT measured by this pilot and is not included.")

    mem = report.get("memory_mb", {})
    if mem:
        print("\n[Memory, MB]")
        for key, value in mem.items():
            print("  %-28s %s" % (key, "n/a" if value is None else "%.0f" % value))

    warns = report.get("warnings", {})
    print("\n[Warnings: %d distinct]" % len(warns))
    for key, count in sorted(warns.items(), key=lambda kv: -kv[1])[:25]:
        print("  x%-5d %s" % (count, key))

    errs = report.get("errors", [])
    print("\n[Errors: %d]" % len(errs))
    for err in errs:
        print("  --- stage: %s" % err["stage"])
        print("  " + err["traceback"].strip().replace("\n", "\n  "))

    print("\nReport saved to: %s" % REPORT_PATH)
    print(line)


# ----------------------------------------------------------------------------
# Main pilot
# ----------------------------------------------------------------------------
def main():
    report = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "design": {
            "cv": "RepeatedStratifiedKFold(n_splits=%d, n_repeats=%d, random_state=%d)" % (N_SPLITS, N_REPEATS, SEED),
            "pilot_split_index": PILOT_SPLIT_INDEX,
            "explainer": "shap.PermutationExplainer",
            "background": "shap.kmeans(X_train, %d).data (centroids snapped to observed values)" % N_BACKGROUND,
            "n_permutations_per_sample": N_PERMUTATIONS,
            "scenario": "none (no augmentation)",
        },
        "timings_sec": {},
        "memory_mb": {},
        "warnings": {},
        "errors": [],
    }
    warn_counts = Counter()
    stage = "environment"

    try:
        # ---- environment -------------------------------------------------
        try:
            import keras
            keras_version = keras.__version__
        except Exception:
            keras_version = None
        notes = []
        if not sys.version.startswith("3.10"):
            notes.append("Python is %s, expected 3.10.x (hcc_thesis)." % sys.version.split()[0])
        if os.environ.get("CONDA_DEFAULT_ENV") != "hcc_thesis":
            notes.append("Active conda env is '%s', expected 'hcc_thesis'." % os.environ.get("CONDA_DEFAULT_ENV"))
        if _determinism_error:
            notes.append("enable_op_determinism() failed: %s" % _determinism_error)
        report["environment"] = {
            "python": sys.version.split()[0],
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "keras": keras_version,
            "sklearn": sklearn.__version__,
            "shap": shap.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tf_devices": [d.name for d in tf.config.list_physical_devices()],
            "notes": notes,
        }
        for note in notes:
            print("WARNING: " + note)
        report["memory_mb"]["after_imports"] = rss_mb()

        # ---- data + the ONE fold ----------------------------------------
        stage = "data_and_split"
        data_path = DATA_PATH
        if not os.path.exists(data_path):
            data_path = os.path.join(HERE, DATA_PATH)
        df = pd.read_csv(data_path, header=None, na_values="?")
        if df.shape[1] != len(COLUMN_NAMES):
            raise ValueError("Expected %d columns in %s, got %d" % (len(COLUMN_NAMES), data_path, df.shape[1]))
        df.columns = COLUMN_NAMES
        X_raw = df.drop("Class", axis=1)
        y = df["Class"].values.astype(int)
        feature_names_all = list(X_raw.columns)

        rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
        splits = list(rskf.split(X_raw, y))
        train_idx, val_idx = splits[PILOT_SPLIT_INDEX]
        X_tr_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Preprocessing fit on the fold's TRAIN part only (no leakage).
        seeded_mi = partial(mutual_info_classif, random_state=SEED)
        prep = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
            ("selector", SelectKBest(seeded_mi, k=N_FEATURES)),
        ])
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_val = prep.transform(X_val_raw)
        selected_idx = prep.named_steps["selector"].get_support(indices=True)
        selected_features = [feature_names_all[i] for i in selected_idx]

        report["fold"] = {
            "split_index": PILOT_SPLIT_INDEX,
            "n_total_splits": len(splits),
            "n_train": int(len(y_tr)),
            "n_dies_train": int((y_tr == CLASS_DIES).sum()),
            "n_val": int(len(y_val)),
            "n_dies_val": int((y_val == CLASS_DIES).sum()),
            "n_selected_features": len(selected_features),
            "selected_features": selected_features,
        }
        print("Fold: train=%d (Dies=%d)  out-of-fold=%d (Dies=%d)" % (
            len(y_tr), (y_tr == CLASS_DIES).sum(), len(y_val), (y_val == CLASS_DIES).sum()))

        # ---- full training of SAE + GRU + DBN ---------------------------
        stage = "train_sae"
        with record_warnings(warn_counts, "train_sae"), timed(report, "train_sae"):
            encoder = train_sae(X_tr)
        X_tr_sae = encoder.predict(X_tr, verbose=0)
        X_val_sae = encoder.predict(X_val, verbose=0)

        stage = "train_gru"
        with record_warnings(warn_counts, "train_gru"), timed(report, "train_gru"):
            gru_model = train_gru(X_tr_sae, y_tr)

        stage = "train_dbn"
        with record_warnings(warn_counts, "train_dbn"), timed(report, "train_dbn"):
            dbn_model = train_dbn(X_tr_sae, y_tr)

        gru_tr = predict_gru(gru_model, X_tr_sae)
        dbn_tr = predict_dbn(dbn_model, X_tr_sae)
        w_gru = accuracy_score(y_tr, (gru_tr >= 0.5).astype(int))
        w_dbn = accuracy_score(y_tr, (dbn_tr >= 0.5).astype(int))
        report["fold"]["w_gru"] = float(w_gru)
        report["fold"]["w_dbn"] = float(w_dbn)
        report["memory_mb"]["after_training"] = rss_mb()
        report["memory_mb"]["peak_after_training"] = peak_rss_mb()

        # ---- wrapper + equivalence check --------------------------------
        stage = "wrapper_check"
        model = FinalModel(encoder, gru_model, dbn_model, w_gru, w_dbn)
        ref_gru = predict_gru(gru_model, X_val_sae)
        ref_dbn = predict_dbn(dbn_model, X_val_sae)
        ref_final = (w_gru * ref_gru + w_dbn * ref_dbn) / (w_gru + w_dbn)
        wrapped_final = model.predict_final_proba(X_val)
        max_diff = float(np.max(np.abs(ref_final - wrapped_final)))
        ref_pred = (ref_final >= 0.5).astype(int)
        report["wrapper_check"] = {
            "max_abs_diff": max_diff,
            "ok": bool(max_diff < 1e-4),
            "fold_accuracy": float(accuracy_score(y_val, ref_pred)),
            "fold_mcc": float(matthews_corrcoef(y_val, ref_pred)),
            "fold_recall_dies": float(recall_score(y_val, ref_pred, pos_label=CLASS_DIES, zero_division=0)),
        }
        if not report["wrapper_check"]["ok"]:
            raise RuntimeError(
                "predict_final_proba does not match the models.py prediction path "
                "(max abs diff %.3e); refusing to explain a different function." % max_diff)

        # ---- SHAP: PermutationExplainer ---------------------------------
        stage = "shap_setup"
        n_feat = X_tr.shape[1]
        max_evals = N_PERMUTATIONS * (2 * n_feat + 1)  # -> exactly N_PERMUTATIONS permutations/sample
        background = shap.kmeans(X_tr, N_BACKGROUND).data
        masker = shap.maskers.Independent(background, max_samples=N_BACKGROUND)
        try:
            explainer = shap.PermutationExplainer(
                model.predict_final_proba, masker, feature_names=selected_features, seed=SEED)
        except TypeError:  # older SHAP without the `seed` argument
            explainer = shap.PermutationExplainer(
                model.predict_final_proba, masker, feature_names=selected_features)

        report["shap"] = {"max_evals_per_sample": max_evals, "n_background": int(background.shape[0])}

        # Throwaway call so one-off TensorFlow start-up cost is not billed to the projection.
        model.predict_final_proba(X_val[:2])

        stage = "shap_warmup"
        model.reset_counters()
        t0 = time.perf_counter()
        with record_warnings(warn_counts, "shap_warmup"):
            explainer(X_val[:WARMUP_SAMPLES], max_evals=max_evals, silent=True)
        warm_seconds = time.perf_counter() - t0
        per_sample = warm_seconds / WARMUP_SAMPLES
        projected_fold = per_sample * len(X_val)
        report["shap"]["warmup"] = {
            "n_samples": WARMUP_SAMPLES,
            "seconds": warm_seconds,
            "sec_per_sample": per_sample,
            "model_calls": model.n_calls,
            "model_rows": model.n_rows,
            "projected_fold_seconds": projected_fold,
        }
        print("SHAP warm-up: %.2f s for %d samples -> projected %s for the fold" % (
            warm_seconds, WARMUP_SAMPLES, fmt_min(projected_fold)))

        if projected_fold / 60.0 > ABORT_IF_FOLD_SHAP_EXCEEDS_MIN:
            report["shap"]["full_fold_skipped"] = (
                "projected %.1f min exceeds PILOT_MAX_SHAP_MIN=%.0f; full-fold SHAP not run. "
                "Re-run with a larger PILOT_MAX_SHAP_MIN to override." % (
                    projected_fold / 60.0, ABORT_IF_FOLD_SHAP_EXCEEDS_MIN))
            print("Full-fold SHAP skipped: " + report["shap"]["full_fold_skipped"])
        else:
            stage = "shap_full_fold"
            model.reset_counters()
            t0 = time.perf_counter()
            with record_warnings(warn_counts, "shap_full_fold"):
                explanation = explainer(X_val, max_evals=max_evals, silent=True)
            shap_seconds = time.perf_counter() - t0
            calls, rows = model.n_calls, model.n_rows  # capture before any extra model call
            report["timings_sec"]["shap_full_fold"] = shap_seconds
            report["memory_mb"]["after_shap"] = rss_mb()

            values = np.asarray(explanation.values)
            base = np.asarray(explanation.base_values).reshape(-1)
            f_x = model.predict_final_proba(X_val)
            additivity_err = float(np.max(np.abs(values.sum(axis=1) + base - f_x)))
            mean_abs = np.abs(values).mean(axis=0)
            top5 = [selected_features[i] for i in np.argsort(-mean_abs)[:5]]

            report["shap"]["full_fold"] = {
                "n_samples": int(len(X_val)),
                "seconds": shap_seconds,
                "sec_per_sample": shap_seconds / len(X_val),
                "model_calls": int(calls),
                "model_rows": int(rows),
                "theoretical_masked_evals": int(len(X_val) * max_evals),
                "additivity_max_abs_err": additivity_err,
                "shap_values_shape": list(values.shape),
                "top5_features": top5,
            }

            train_seconds = sum(report["timings_sec"].get(k, 0.0) for k in ("train_sae", "train_gru", "train_dbn"))
            per_fold = train_seconds + shap_seconds
            report["projection"] = {
                "per_fold_seconds": per_fold,
                "none_25_folds_seconds": 25 * per_fold,
                "two_scenarios_seconds": 50 * per_fold,
                "excludes": "CTGAN training time (not measured by this pilot)",
            }

    except Exception:
        report["errors"].append({"stage": stage, "traceback": traceback.format_exc()})

    finally:
        report["memory_mb"]["final"] = rss_mb()
        report["memory_mb"]["peak_overall"] = peak_rss_mb()
        report["warnings"] = dict(warn_counts)
        report["finished_at"] = datetime.now().isoformat(timespec="seconds")
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print_summary(report)


if __name__ == "__main__":
    main()
