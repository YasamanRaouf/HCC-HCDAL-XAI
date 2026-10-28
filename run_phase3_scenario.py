"""
run_phase3_scenario.py

Full Phase 3 run for ONE scenario ("none" or "ctgan"), across all 25 folds
of the approved design: RepeatedStratifiedKFold(n_splits=5, n_repeats=5).
This is the scaled-up version of pilot_phase3.py -- same preprocessing,
same FinalModel wrapper, same PermutationExplainer configuration, same
wrapper-equivalence check per fold -- nothing about the approved Design B
or the Stability amendment is changed here.

For scenario="ctgan", augmentation.augment_ctgan() (already written and
validated in Phase 2) is applied to the fold's TRAIN part only, after the
preprocessing pipeline is fit -- never to the out-of-fold (validation) part.
No other project file or core logic is modified.

Resilience: results are appended fold-by-fold to disk as soon as each fold
finishes, and the script SKIPS any split index already present in the
output files on start-up. If the run is interrupted (Ctrl-C, sleep, crash),
re-running the same command resumes from the next incomplete fold instead
of starting over.

USAGE (from the project folder, hcc_thesis environment, Python 3.10.20):
    conda activate hcc_thesis
    cd /Users/yasaman/master_thesis_HCC/HepatocellularCarcinomaDet
    python run_phase3_scenario.py none
    python run_phase3_scenario.py ctgan

Outputs (under results/):
    phase3_{scenario}_folds.csv       one row per fold: perf + timings +
                                       Consistency (Pearson) + wrapper check
    phase3_{scenario}_rankings.jsonl  one line per fold: full mean(|SHAP|)
                                       feature ranking (for Generalizability)
    phase3_{scenario}_shap_oof.jsonl  one line per out-of-fold sample per
                                       repeat (for pooled Stability)
"""

import os
os.environ.setdefault("PYTHONHASHSEED", "0")

import sys
import json
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as exc:
    print("WARNING: enable_op_determinism() failed: %r" % exc)

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score
from scipy.stats import pearsonr

from data_loader import COLUMN_NAMES
from models import train_sae, train_gru, predict_gru, train_dbn, predict_dbn
from augmentation import augment_ctgan
from phase3_common import (
    build_fold_preprocessor, FinalModel, check_wrapper_matches_reference,
    make_permutation_explainer, explain_fold, permutation_feature_importance,
)

N_SPLITS = 5
N_REPEATS = 5
CLASS_DIES = 0
CTGAN_EPOCHS = 300  # same value used for the Phase 2 single-run comparison

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get("HCC_DATA_PATH", os.path.join(HERE, "Data", "hcc-data.csv"))
RESULTS_DIR = os.path.join(HERE, "results")

# Fixed schema for phase3_{scenario}_folds.csv. Every row written (success
# OR failure) is reindexed to EXACTLY these columns, in this order, so that
# appending a successful fold's row (many columns) after a failed fold's
# row (few columns), or vice versa, can never shift columns or corrupt the
# CSV. "status" ("ok"/"failed") is the single source of truth for whether a
# fold actually completed -- see already_done_splits().
CANONICAL_FOLD_COLUMNS = [
    "split_index", "repeat", "fold", "scenario", "status", "error",
    "n_train", "n_dies_train", "n_val", "n_dies_val", "n_selected_features",
    "train_sae_sec", "train_gru_sec", "train_dbn_sec",
    "w_gru", "w_dbn", "wrapper_ok", "wrapper_max_diff",
    "accuracy", "mcc", "recall_dies", "shap_sec",
    "consistency_pearson_r", "consistency_pearson_p",
    "fold_wall_sec",
]


def load_raw_dataset():
    df = pd.read_csv(DATA_PATH, header=None, na_values="?")
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError("Expected %d columns in %s, got %d" % (
            len(COLUMN_NAMES), DATA_PATH, df.shape[1]))
    df.columns = COLUMN_NAMES
    X_raw = df.drop("Class", axis=1)
    y = df["Class"].values.astype(int)
    return X_raw, y, list(X_raw.columns)


def build_global_feature_embedding(X_raw):
    """
    A SINGLE impute+scale embedding (NO feature selection) fit on all 165
    patients, used ONLY to find each out-of-fold sample's nearest
    same-class neighbor for the Stability metric -- never for training or
    for any accuracy/MCC/recall number.

    Why this is not a leakage concern: it does not touch model training or
    evaluation at all; it is a fixed, one-time geometric description of
    "how similar are these two patients" used purely to pick WHICH pairs
    to compare post hoc, after every fold's predictions/SHAP values are
    already final. Using each fold's own selected-30-features space instead
    would make "clinically similar" depend on which features a particular
    fold happened to pick, which is a property of the model, not the
    patient -- the whole point of Stability is to ask "for two similar
    PATIENTS, are the explanations similar", so the similarity metric
    should not itself be fold-dependent.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    return scaler.fit_transform(imputer.fit_transform(X_raw))  # shape (165, 49)


def embed_shap_in_full_feature_space(shap_values_row, selected_features, all_feature_names):
    """Map a fold's SHAP vector (length = n_selected, e.g. 30) into the
    full 49-feature space, with 0 for features not selected by that fold.
    Makes SHAP vectors from different folds (which may select different
    feature subsets) directly comparable for the Stability distance."""
    full = np.zeros(len(all_feature_names), dtype=float)
    idx_map = {name: i for i, name in enumerate(all_feature_names)}
    for val, feat in zip(shap_values_row, selected_features):
        full[idx_map[feat]] = val
    return full


def already_done_splits(folds_csv_path):
    """Only a fold whose row has status == 'ok' counts as done. Folds that
    previously errored out or failed the wrapper-equivalence check are NOT
    considered done, so the next run retries them instead of silently
    skipping them forever."""
    if not os.path.exists(folds_csv_path):
        return set()
    try:
        done = pd.read_csv(folds_csv_path)
        if "status" not in done.columns:
            return set()  # older/foreign file without a status column -- retry everything rather than guess
        ok_rows = done[done["status"] == "ok"]
        return set(ok_rows["split_index"].astype(int).tolist())
    except Exception:
        return set()


def append_row_csv(path, row_dict):
    """Always reindex to CANONICAL_FOLD_COLUMNS before writing, so a
    success row (many fields) and a failure row (few fields) are written
    with IDENTICAL columns in IDENTICAL order -- appending can never
    misalign the CSV regardless of which rows came before it."""
    df_row = pd.DataFrame([row_dict]).reindex(columns=CANONICAL_FOLD_COLUMNS)
    header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=header, index=False)


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def run_one_fold(scenario, split_index, repeat_index, fold_index,
                  X_tr_raw, X_val_raw, y_tr, y_val, orig_val_indices, feature_names_all,
                  global_embedding):
    fold_report = {"split_index": split_index, "repeat": repeat_index, "fold": fold_index,
                    "scenario": scenario}
    t_fold_start = time.perf_counter()

    prep = build_fold_preprocessor(seed=SEED)
    X_tr = prep.fit_transform(X_tr_raw, y_tr)
    X_val = prep.transform(X_val_raw)
    selected_idx = prep.named_steps["selector"].get_support(indices=True)
    selected_features = [feature_names_all[i] for i in selected_idx]

    y_tr_used = y_tr
    if scenario == "ctgan":
        X_tr, y_tr_used = augment_ctgan(X_tr, y_tr, selected_features, epochs=CTGAN_EPOCHS, random_state=SEED)
    elif scenario != "none":
        raise ValueError("Unknown scenario: %s" % scenario)

    fold_report["n_train"] = int(len(y_tr_used))
    fold_report["n_dies_train"] = int((y_tr_used == CLASS_DIES).sum())
    fold_report["n_val"] = int(len(y_val))
    fold_report["n_dies_val"] = int((y_val == CLASS_DIES).sum())
    fold_report["n_selected_features"] = len(selected_features)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # keras input_shape UserWarning etc., already characterized in the pilot
        t0 = time.perf_counter()
        encoder = train_sae(X_tr)
        fold_report["train_sae_sec"] = time.perf_counter() - t0

        X_tr_sae = encoder.predict(X_tr, verbose=0)
        X_val_sae = encoder.predict(X_val, verbose=0)

        t0 = time.perf_counter()
        gru_model = train_gru(X_tr_sae, y_tr_used)
        fold_report["train_gru_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dbn_model = train_dbn(X_tr_sae, y_tr_used)
        fold_report["train_dbn_sec"] = time.perf_counter() - t0

    gru_tr = predict_gru(gru_model, X_tr_sae)
    dbn_tr = predict_dbn(dbn_model, X_tr_sae)
    w_gru = accuracy_score(y_tr_used, (gru_tr >= 0.5).astype(int))
    w_dbn = accuracy_score(y_tr_used, (dbn_tr >= 0.5).astype(int))
    fold_report["w_gru"] = float(w_gru)
    fold_report["w_dbn"] = float(w_dbn)

    model = FinalModel(encoder, gru_model, dbn_model, w_gru, w_dbn)

    ref_gru = predict_gru(gru_model, X_val_sae)
    ref_dbn = predict_dbn(dbn_model, X_val_sae)
    ref_final = (w_gru * ref_gru + w_dbn * ref_dbn) / (w_gru + w_dbn)
    wrapper_ok, wrapper_max_diff = check_wrapper_matches_reference(model, X_val, ref_final)
    fold_report["wrapper_ok"] = bool(wrapper_ok)
    fold_report["wrapper_max_diff"] = wrapper_max_diff
    if not wrapper_ok:
        fold_report["status"] = "failed"
        fold_report["error"] = "wrapper mismatch (max diff %.3e); fold skipped for SHAP/PFI" % wrapper_max_diff
        fold_report["fold_wall_sec"] = time.perf_counter() - t_fold_start
        return fold_report, None, None

    pred = (ref_final >= 0.5).astype(int)
    fold_report["accuracy"] = float(accuracy_score(y_val, pred))
    fold_report["mcc"] = float(matthews_corrcoef(y_val, pred))
    fold_report["recall_dies"] = float(recall_score(y_val, pred, pos_label=CLASS_DIES, zero_division=0))

    explainer, max_evals = make_permutation_explainer(model, X_tr, selected_features)
    shap_values, base_values, shap_sec = explain_fold(explainer, X_val, max_evals)
    fold_report["shap_sec"] = shap_sec

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking_record = {
        "split_index": split_index, "repeat": repeat_index, "fold": fold_index,
        "scenario": scenario, "selected_features": selected_features,
        "mean_abs_shap": mean_abs_shap,
    }

    pfi = permutation_feature_importance(model, X_val, y_val)
    # Consistency = Pearson correlation between SHAP's mean(|value|) ranking
    # and the pipeline's own Permutation Feature Importance (approved
    # design; see phase3_common.permutation_feature_importance docstring).
    if np.std(pfi) > 0 and np.std(mean_abs_shap) > 0:
        consistency_r, consistency_p = pearsonr(mean_abs_shap, pfi)
    else:
        consistency_r, consistency_p = float("nan"), float("nan")
    fold_report["consistency_pearson_r"] = float(consistency_r)
    fold_report["consistency_pearson_p"] = float(consistency_p)

    oof_records = []
    for i, orig_idx in enumerate(orig_val_indices):
        shap_full49 = embed_shap_in_full_feature_space(
            shap_values[i], selected_features, feature_names_all)
        oof_records.append({
            "split_index": split_index, "repeat": repeat_index, "fold": fold_index,
            "scenario": scenario, "orig_sample_index": int(orig_idx),
            "true_class": int(y_val[i]), "pred_proba": float(ref_final[i]),
            "selected_features": selected_features,
            "shap_values": shap_values[i],                 # raw, fold's own 30-dim space (for reference/debugging)
            "shap_values_full49": shap_full49,              # comparable across folds -- used for Stability
            "global_similarity_vector": global_embedding[orig_idx],  # fixed, fold-independent -- used to find neighbors
        })

    fold_report["fold_wall_sec"] = time.perf_counter() - t_fold_start
    return fold_report, ranking_record, oof_records


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("none", "ctgan"):
        print("Usage: python run_phase3_scenario.py [none|ctgan]")
        sys.exit(1)
    scenario = sys.argv[1]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    folds_csv = os.path.join(RESULTS_DIR, "phase3_%s_folds.csv" % scenario)
    rankings_jsonl = os.path.join(RESULTS_DIR, "phase3_%s_rankings.jsonl" % scenario)
    shap_oof_jsonl = os.path.join(RESULTS_DIR, "phase3_%s_shap_oof.jsonl" % scenario)

    X_raw, y, feature_names_all = load_raw_dataset()
    global_embedding = build_global_feature_embedding(X_raw)  # fixed, used only for Stability's neighbor-finding
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rskf.split(X_raw, y))
    total = len(splits)

    done = already_done_splits(folds_csv)
    if done:
        print("Resuming: %d/%d folds already completed, skipping them." % (len(done), total))

    for split_index, (train_idx, val_idx) in enumerate(splits):
        if split_index in done:
            continue
        repeat_index, fold_index = divmod(split_index, N_SPLITS)

        print("\n[%s] split %d/%d  (repeat %d, fold %d)" % (
            scenario, split_index + 1, total, repeat_index, fold_index))
        t0 = time.perf_counter()
        try:
            X_tr_raw = X_raw.iloc[train_idx]
            X_val_raw = X_raw.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_report, ranking_record, oof_records = run_one_fold(
                scenario, split_index, repeat_index, fold_index,
                X_tr_raw, X_val_raw, y_tr, y_val, val_idx, feature_names_all,
                global_embedding)

            append_row_csv(folds_csv, fold_report)
            if ranking_record is not None:
                append_jsonl(rankings_jsonl, ranking_record)
            if oof_records:
                for rec in oof_records:
                    append_jsonl(shap_oof_jsonl, rec)

            status = "OK" if fold_report.get("wrapper_ok", False) else "SKIPPED (wrapper mismatch)"
            print("  %s in %.1f s" % (status, time.perf_counter() - t0))
            if "accuracy" in fold_report:
                print("  acc=%.3f  MCC=%.3f  recall(Dies)=%.3f  consistency_r=%.3f  shap=%.1fs" % (
                    fold_report["accuracy"], fold_report["mcc"], fold_report["recall_dies"],
                    fold_report["consistency_pearson_r"], fold_report["shap_sec"]))

        except Exception:
            tb = traceback.format_exc()
            print("  ERROR on split %d:\n%s" % (split_index, tb))
            append_row_csv(folds_csv, {
                "split_index": split_index, "repeat": repeat_index, "fold": fold_index,
                "scenario": scenario, "error": tb.strip().splitlines()[-1],
            })
            # Continue to the next fold rather than aborting the whole run.
            continue

    print("\nScenario '%s' complete (or already was). Results in:\n  %s\n  %s\n  %s" % (
        scenario, folds_csv, rankings_jsonl, shap_oof_jsonl))


if __name__ == "__main__":
    main()
