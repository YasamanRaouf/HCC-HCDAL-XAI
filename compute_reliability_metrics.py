"""
compute_reliability_metrics.py

Phase 3 -- reliability analysis (Generalizability, Consistency, Stability)
for the two scenarios written by run_phase3_scenario.py:

    none   no augmentation
    ctgan  CTGAN augmentation of every fold's training part

This script ONLY READS the existing result files and computes DESCRIPTIVE
summaries. It trains nothing, explains nothing, applies NO hypothesis test
and reports NO p-value anywhere. (A none-vs-ctgan statistical comparison is
deliberately out of scope here.)

INPUTS, per scenario, in --results-dir (default: <this folder>/results):
    phase3_{scenario}_folds.csv        one row per fold
    phase3_{scenario}_rankings.jsonl   one line per fold: mean(|SHAP|) over the
                                       fold's selected features
    phase3_{scenario}_shap_oof.jsonl   one line per out-of-fold patient per repeat

STRUCTURE FACTS (taken from run_phase3_scenario.py, re-validated on load):
  * valid fold      = a folds.csv row with non-null `accuracy`. The `status`
                      column is NaN in the current CSVs and is NOT used.
  * patient id      = `orig_sample_index`, the 0-based row position of the
                      patient in Data/hcc-data.csv.
  * class coding    = `true_class`: 0 = Dies (minority), 1 = Lives.
  * 49-feature space = [c for c in data_loader.COLUMN_NAMES if c != "Class"],
                      the order used when `shap_values_full49` was written;
                      features NOT selected by a fold hold 0.
  * CV layout       = 5 repeats x 5 folds = 25 splits;
                      repeat = split_index // 5, fold = split_index % 5.
                      Inside one repeat every patient is out-of-fold exactly once.

STABILITY (approved design, descriptive only)
  * Neighbor search happens ONLY inside the same repeat, among patients of the
    SAME class; a patient is never its own neighbor.
  * Nearest neighbor = smallest Euclidean distance in the fixed 49-feature
    `global_similarity_vector` (ties -> lowest patient id).
  * Per patient: Euclidean distance between the two patients' 49-dim SHAP
    vectors (`shap_values_full49`).
  * Per repeat: MEDIAN over patients; then mean +/- sample SD (ddof=1) of the
    5 repeat medians.
  * Random reference: same constraints (same repeat, same class, not self), but
    the neighbor is drawn uniformly at random with a fixed seed. The generator
    is seeded per repeat only and patients are visited in ascending id order, so
    BOTH scenarios get exactly the same random pairs.
  * Per-patient distances are saved for boxplots only. They are NOT independent
    observations and no test is applied to them.

GENERALIZABILITY
  * Each fold's mean(|SHAP|) is mapped into the 49-feature space (0 for
    unselected features), converted to average ranks (ties share a rank), and
    compared between folds with Spearman's rho (= Pearson correlation of those
    ranks). Summaries are given over all 300 fold pairs, within-repeat pairs
    and cross-repeat pairs, plus the 5 within-repeat means (one per repeat).

CONSISTENCY
  * Taken as-is from the existing `consistency_pearson_r` column (Pearson r
    between mean(|SHAP|) and the pipeline's own permutation feature importance,
    per fold). Negative values are kept. `consistency_pearson_p` is not used.

USAGE (hcc_thesis environment, project folder):
    python compute_reliability_metrics.py
    python compute_reliability_metrics.py --results-dir results
    python compute_reliability_metrics.py --overwrite     # only if outputs exist

By default the script REFUSES to run if any of its output files already exists.

OUTPUTS (written into --results-dir):
    phase3_reliability_per_repeat.csv
    phase3_reliability_summary.csv
    phase3_stability_patient_distances.csv
    phase3_generalizability_pairs.csv
    phase3_reliability_report.json

Exit codes: 0 = ok, 1 = input data not in the expected structure (nothing is
written), 2 = refused to overwrite existing outputs.
"""

import os
import sys
import json
import argparse
import itertools
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata  # ranking helper only; no hypothesis test is used in this file

from data_loader import COLUMN_NAMES


HERE = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = ["none", "ctgan"]
N_REPEATS = 5   # must match run_phase3_scenario.py
N_SPLITS = 5    # folds per repeat; must match run_phase3_scenario.py
CLASS_DIES = 0
CLASS_LIVES = 1
RANDOM_REFERENCE_SEED = 42

ARRAY_TOL = 1e-9             # consistency of shap_values vs shap_values_full49
SIM_TOL = 1e-12              # a patient's similarity vector must be identical everywhere
RANK_CROSSCHECK_RTOL = 1e-5  # rankings.jsonl vs values recomputed from shap_oof.jsonl
RANK_CROSSCHECK_ATOL = 1e-7

ALL_FEATURES = [c for c in COLUMN_NAMES if c != "Class"]
N_ALL_FEATURES = len(ALL_FEATURES)
FEATURE_INDEX = dict((name, i) for i, name in enumerate(ALL_FEATURES))

OUTPUT_FILES = {
    "per_repeat": "phase3_reliability_per_repeat.csv",
    "summary": "phase3_reliability_summary.csv",
    "patient_distances": "phase3_stability_patient_distances.csv",
    "generalizability_pairs": "phase3_generalizability_pairs.csv",
    "report": "phase3_reliability_report.json",
}

LIMITATIONS = [
    "The 5 repeats are NOT independent: every repeat re-partitions the same 165 "
    "patients, each patient is out-of-fold once per repeat, and training sets "
    "overlap heavily across folds and repeats. The SD over the 5 repeat-level "
    "values only describes variation between re-partitionings; it is not a "
    "standard error or a confidence interval, and n = 5 is small.",
    "Per-patient distances are saved for descriptive plots only. They are not "
    "independent observations (a patient appears in every repeat, one patient "
    "can be the neighbor of several patients, and distances within a repeat "
    "share fold models). No hypothesis test or p-value is computed on them.",
    "Stability neighbors are searched inside a repeat but across folds, as in "
    "the approved design: a patient and its neighbor may be explained by "
    "DIFFERENT fold models, so SHAP distance mixes patient dissimilarity with "
    "model-to-model variation. The fraction of same-fold pairs is reported "
    "(frac_*_same_fold) for both the nearest and the random reference.",
    "The distances are proximity in a fixed 49-feature min-max-scaled space "
    "(global-mean imputation, all features weighted equally, binary and "
    "continuous mixed, no clinical weighting, no clinician validation) and "
    "Euclidean distance between SHAP vectors (unselected features = 0). They "
    "are statistical proxies, NOT established clinical similarity. The "
    "similarity embedding was fit on all 165 patients; it is used only to pick "
    "which pairs to compare after the fact and never affected training or any "
    "performance number.",
    "Generalizability compares fold-level mean(|SHAP|) rankings with ties "
    "(unselected features share the zero rank). A fold takes part in 24 pairs "
    "and folds share most of their training data, so pair values are not "
    "independent; their spread is descriptive only.",
    "Consistency is a per-fold Pearson r between mean(|SHAP|) and permutation "
    "feature importance computed on only about 33 out-of-fold patients (about "
    "12 Dies), so it is noisy. Negative values are kept. Fold values are not "
    "independent.",
    "The `status` column of the current folds CSVs is NaN; valid folds are "
    "identified by non-null `accuracy`.",
    "Random-reference pairs use a fixed seed (%d) per repeat and are identical "
    "in both scenarios, so the nearest-vs-random contrast is the same "
    "construction in both scenarios." % RANDOM_REFERENCE_SEED,
]


class DataIntegrityError(Exception):
    """The input files are not in the expected structure. The script stops
    instead of guessing and writes nothing."""


# ----------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------
def read_jsonl(path):
    if not os.path.exists(path):
        raise DataIntegrityError("Missing input file: %s" % path)
    records = []
    with open(path, "r") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise DataIntegrityError("%s, line %d: invalid JSON (%s)" % (path, line_no, exc))
    return records


def require_keys(record, keys, where):
    missing = [k for k in keys if k not in record]
    if missing:
        raise DataIntegrityError("%s: a record is missing key(s) %s" % (where, missing))


def as_float_vector(values, expected_len, where):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise DataIntegrityError("%s: not a numeric list" % where)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise DataIntegrityError(
            "%s: expected a vector of length %d, got shape %s" % (where, expected_len, arr.shape))
    if not np.all(np.isfinite(arr)):
        raise DataIntegrityError("%s: contains NaN or inf" % where)
    return arr


def load_valid_folds(path, scenario):
    if not os.path.exists(path):
        raise DataIntegrityError("Missing input file: %s" % path)
    df = pd.read_csv(path)
    required = ["split_index", "repeat", "fold", "scenario", "n_val",
                "n_selected_features", "accuracy", "consistency_pearson_r"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise DataIntegrityError("%s: missing column(s) %s" % (path, missing_cols))

    n_rows = int(len(df))
    valid = df[df["accuracy"].notna()].copy()
    n_invalid = n_rows - int(len(valid))

    for col in ["split_index", "repeat", "fold", "n_val", "n_selected_features"]:
        if valid[col].isna().any():
            raise DataIntegrityError("%s: valid rows have empty `%s`" % (path, col))
        valid[col] = valid[col].astype(int)

    if valid["split_index"].duplicated().any():
        dup = sorted(valid.loc[valid["split_index"].duplicated(keep=False), "split_index"].unique().tolist())
        raise DataIntegrityError(
            "%s: split_index appears in more than one valid row: %s. "
            "The file probably contains a re-run appended on top of an earlier run." % (path, dup))

    expected_ids = set(range(N_REPEATS * N_SPLITS))
    found_ids = set(valid["split_index"].tolist())
    if found_ids != expected_ids:
        raise DataIntegrityError(
            "%s: expected valid folds %d..%d. Missing: %s. Unexpected: %s." % (
                path, 0, N_REPEATS * N_SPLITS - 1,
                sorted(expected_ids - found_ids), sorted(found_ids - expected_ids)))

    if not (valid["repeat"] == valid["split_index"] // N_SPLITS).all():
        raise DataIntegrityError("%s: `repeat` != split_index // %d for some rows" % (path, N_SPLITS))
    if not (valid["fold"] == valid["split_index"] % N_SPLITS).all():
        raise DataIntegrityError("%s: `fold` != split_index %% %d for some rows" % (path, N_SPLITS))
    if not (valid["scenario"] == scenario).all():
        raise DataIntegrityError("%s: `scenario` column is not '%s' for all valid rows" % (path, scenario))

    valid = valid.sort_values("split_index").reset_index(drop=True)
    return valid, n_rows, n_invalid


def load_rankings(path, scenario, valid_folds):
    records = read_jsonl(path)
    lookup = dict((int(row.split_index), row) for row in valid_folds.itertuples())
    by_split = {}
    n_excluded = 0
    for rec in records:
        require_keys(rec, ["split_index", "repeat", "fold", "scenario",
                           "selected_features", "mean_abs_shap"], path)
        if rec["scenario"] != scenario:
            raise DataIntegrityError("%s: record with scenario '%s', expected '%s'" % (
                path, rec["scenario"], scenario))
        sid = int(rec["split_index"])
        if sid not in lookup:
            n_excluded += 1
            continue
        if sid in by_split:
            raise DataIntegrityError("%s: split_index %d appears more than once" % (path, sid))
        row = lookup[sid]
        if int(rec["repeat"]) != int(row.repeat) or int(rec["fold"]) != int(row.fold):
            raise DataIntegrityError("%s: repeat/fold of split %d disagree with the folds CSV" % (path, sid))
        feats = list(rec["selected_features"])
        if len(feats) != int(row.n_selected_features):
            raise DataIntegrityError("%s: split %d has %d selected features, folds CSV says %d" % (
                path, sid, len(feats), int(row.n_selected_features)))
        if len(set(feats)) != len(feats):
            raise DataIntegrityError("%s: split %d has duplicate selected features" % (path, sid))
        unknown = [f for f in feats if f not in FEATURE_INDEX]
        if unknown:
            raise DataIntegrityError("%s: split %d has unknown feature name(s) %s" % (path, sid, unknown))
        vals = as_float_vector(rec["mean_abs_shap"], len(feats), "%s split %d mean_abs_shap" % (path, sid))
        if np.any(vals < 0):
            raise DataIntegrityError("%s: split %d has negative mean_abs_shap" % (path, sid))
        by_split[sid] = {"repeat": int(row.repeat), "fold": int(row.fold),
                         "features": feats, "vals": vals}
    missing = sorted(set(lookup) - set(by_split))
    if missing:
        raise DataIntegrityError("%s: no ranking record for valid fold(s) %s" % (path, missing))
    return by_split, len(records), n_excluded


def load_oof(path, scenario, valid_folds, rankings):
    records = read_jsonl(path)
    lookup = dict((int(row.split_index), row) for row in valid_folds.itertuples())
    needed = ["split_index", "repeat", "fold", "scenario", "orig_sample_index",
              "true_class", "pred_proba", "selected_features", "shap_values",
              "shap_values_full49", "global_similarity_vector"]
    kept = []
    seen = set()
    n_excluded = 0
    for rec in records:
        require_keys(rec, needed, path)
        if rec["scenario"] != scenario:
            raise DataIntegrityError("%s: record with scenario '%s', expected '%s'" % (
                path, rec["scenario"], scenario))
        sid = int(rec["split_index"])
        if sid not in lookup:
            n_excluded += 1
            continue
        row = lookup[sid]
        if int(rec["repeat"]) != int(row.repeat) or int(rec["fold"]) != int(row.fold):
            raise DataIntegrityError("%s: repeat/fold of split %d disagree with the folds CSV" % (path, sid))
        pid = int(rec["orig_sample_index"])
        if (sid, pid) in seen:
            raise DataIntegrityError("%s: patient %d appears twice in split %d" % (path, pid, sid))
        seen.add((sid, pid))
        cls = int(rec["true_class"])
        if cls not in (CLASS_DIES, CLASS_LIVES):
            raise DataIntegrityError("%s: true_class=%d is neither 0 (Dies) nor 1 (Lives)" % (path, cls))
        feats = list(rec["selected_features"])
        if feats != rankings[sid]["features"]:
            raise DataIntegrityError("%s: selected_features of patient %d / split %d differ from rankings.jsonl" % (
                path, pid, sid))
        where = "%s split %d patient %d" % (path, sid, pid)
        shap30 = as_float_vector(rec["shap_values"], len(feats), where + " shap_values")
        shap49 = as_float_vector(rec["shap_values_full49"], N_ALL_FEATURES, where + " shap_values_full49")
        sim = as_float_vector(rec["global_similarity_vector"], N_ALL_FEATURES, where + " global_similarity_vector")
        rebuilt = np.zeros(N_ALL_FEATURES)
        for name, val in zip(feats, shap30):
            rebuilt[FEATURE_INDEX[name]] = val
        if float(np.max(np.abs(rebuilt - shap49))) > ARRAY_TOL:
            raise DataIntegrityError(
                "%s: shap_values_full49 does not match shap_values mapped through the 49-feature "
                "order (feature-order mismatch?)" % where)
        kept.append({"sid": sid, "repeat": int(row.repeat), "fold": int(row.fold),
                     "pid": pid, "cls": cls, "shap49": shap49, "sim": sim, "shap30": shap30})

    by_fold = {}
    for rec in kept:
        by_fold.setdefault(rec["sid"], []).append(rec)
    for sid, row in lookup.items():
        recs = by_fold.get(sid, [])
        if len(recs) != int(row.n_val):
            raise DataIntegrityError(
                "%s: split %d has %d out-of-fold records, folds CSV says n_val=%d "
                "(incomplete or duplicated write?)" % (path, sid, len(recs), int(row.n_val)))
        recomputed = np.mean([np.abs(r["shap30"]) for r in recs], axis=0)
        if not np.allclose(recomputed, rankings[sid]["vals"],
                           rtol=RANK_CROSSCHECK_RTOL, atol=RANK_CROSSCHECK_ATOL):
            raise DataIntegrityError(
                "%s: mean(|SHAP|) recomputed from shap_oof.jsonl does not match rankings.jsonl "
                "for split %d" % (path, sid))
    return kept, len(records), n_excluded


def check_patient_structure(oof, scenario):
    cls_of = {}
    sim_of = {}
    for rec in oof:
        pid = rec["pid"]
        if pid in cls_of:
            if cls_of[pid] != rec["cls"]:
                raise DataIntegrityError("%s: patient %d has different class labels in different records" % (scenario, pid))
            if float(np.max(np.abs(sim_of[pid] - rec["sim"]))) > SIM_TOL:
                raise DataIntegrityError("%s: patient %d has different similarity vectors in different records" % (scenario, pid))
        else:
            cls_of[pid] = rec["cls"]
            sim_of[pid] = rec["sim"]

    by_repeat = {}
    for rec in oof:
        by_repeat.setdefault(rec["repeat"], []).append(rec["pid"])
    reference = None
    for repeat in range(N_REPEATS):
        pids = sorted(by_repeat.get(repeat, []))
        if len(set(pids)) != len(pids):
            raise DataIntegrityError("%s: repeat %d contains the same patient more than once" % (scenario, repeat))
        if reference is None:
            reference = pids
        elif pids != reference:
            raise DataIntegrityError("%s: repeat %d does not contain the same patients as repeat 0" % (scenario, repeat))
    if not reference:
        raise DataIntegrityError("%s: no out-of-fold records found" % scenario)
    return {"cls": cls_of, "sim": sim_of, "patients": reference}


def check_cross_scenario(structures):
    base = structures[SCENARIOS[0]]
    for scenario in SCENARIOS[1:]:
        other = structures[scenario]
        if other["patients"] != base["patients"]:
            raise DataIntegrityError("Scenarios '%s' and '%s' do not cover the same patients" % (SCENARIOS[0], scenario))
        for pid in base["patients"]:
            if other["cls"][pid] != base["cls"][pid]:
                raise DataIntegrityError("Patient %d has a different class in '%s' and '%s'" % (pid, SCENARIOS[0], scenario))
            if float(np.max(np.abs(other["sim"][pid] - base["sim"][pid]))) > SIM_TOL:
                raise DataIntegrityError("Patient %d has a different similarity vector in '%s' and '%s'" % (pid, SCENARIOS[0], scenario))


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def describe(values):
    """Descriptive statistics; the SD is the SAMPLE standard deviation (ddof=1)."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    stats = {
        "n_total": int(arr.size),
        "n_finite": int(finite.size),
        "n_nonfinite": int(arr.size - finite.size),
        "mean": float("nan"),
        "sd_ddof1": float("nan"),
        "median": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
    }
    if finite.size >= 1:
        stats["mean"] = float(np.mean(finite))
        stats["median"] = float(np.median(finite))
        stats["min"] = float(np.min(finite))
        stats["max"] = float(np.max(finite))
    if finite.size >= 2:
        stats["sd_ddof1"] = float(np.std(finite, ddof=1))
    return stats


def compute_stability(oof, scenario):
    patient_rows = []
    repeat_stats = {}
    for repeat in range(N_REPEATS):
        recs = sorted([r for r in oof if r["repeat"] == repeat], key=lambda r: r["pid"])
        n = len(recs)
        pids = np.array([r["pid"] for r in recs])
        cls = np.array([r["cls"] for r in recs])
        folds = np.array([r["fold"] for r in recs])
        sim = np.vstack([r["sim"] for r in recs])
        shp = np.vstack([r["shap49"] for r in recs])

        diff = sim[:, None, :] - sim[None, :, :]
        feat_dist = np.sqrt(np.sum(diff * diff, axis=2))   # fixed feature-space proximity
        # Seeded per repeat only and visited in ascending patient id: identical random pairs in every scenario.
        rng = np.random.default_rng([RANDOM_REFERENCE_SEED, repeat])

        near_shap, rand_shap, near_feat, rand_feat = [], [], [], []
        near_same, rand_same = [], []
        for i in range(n):
            candidates = np.where((cls == cls[i]) & (np.arange(n) != i))[0]   # same repeat, same class, not self
            if candidates.size == 0:
                raise DataIntegrityError(
                    "%s repeat %d: patient %d has no same-class candidate neighbor" % (scenario, repeat, int(pids[i])))
            j_near = int(candidates[int(np.argmin(feat_dist[i, candidates]))])
            j_rand = int(candidates[int(rng.integers(candidates.size))])

            d_shap_near = float(np.sqrt(np.sum((shp[i] - shp[j_near]) ** 2)))
            d_shap_rand = float(np.sqrt(np.sum((shp[i] - shp[j_rand]) ** 2)))
            d_feat_near = float(feat_dist[i, j_near])
            d_feat_rand = float(feat_dist[i, j_rand])
            same_near = bool(folds[i] == folds[j_near])
            same_rand = bool(folds[i] == folds[j_rand])

            near_shap.append(d_shap_near)
            rand_shap.append(d_shap_rand)
            near_feat.append(d_feat_near)
            rand_feat.append(d_feat_rand)
            near_same.append(same_near)
            rand_same.append(same_rand)

            patient_rows.append({
                "scenario": scenario,
                "repeat": repeat,
                "orig_sample_index": int(pids[i]),
                "true_class": int(cls[i]),
                "fold": int(folds[i]),
                "nearest_neighbor_index": int(pids[j_near]),
                "nearest_neighbor_fold": int(folds[j_near]),
                "nearest_same_fold": same_near,
                "feature_distance_nearest": d_feat_near,
                "shap_distance_nearest": d_shap_near,
                "random_neighbor_index": int(pids[j_rand]),
                "random_neighbor_fold": int(folds[j_rand]),
                "random_same_fold": same_rand,
                "feature_distance_random": d_feat_rand,
                "shap_distance_random": d_shap_rand,
            })

        med_near = float(np.median(near_shap))
        med_rand = float(np.median(rand_shap))
        repeat_stats[repeat] = {
            "n_patients": int(n),
            "n_dies": int(np.sum(cls == CLASS_DIES)),
            "n_lives": int(np.sum(cls == CLASS_LIVES)),
            "stability_median_shap_distance_nearest": med_near,
            "stability_median_shap_distance_random": med_rand,
            "stability_ratio_nearest_over_random": (med_near / med_rand) if med_rand > 0 else float("nan"),
            "stability_median_feature_distance_nearest": float(np.median(near_feat)),
            "stability_median_feature_distance_random": float(np.median(rand_feat)),
            "stability_frac_nearest_same_fold": float(np.mean(near_same)),
            "stability_frac_random_same_fold": float(np.mean(rand_same)),
        }
    return patient_rows, repeat_stats


def compute_generalizability(rankings, scenario):
    sids = sorted(rankings)
    mat = np.zeros((len(sids), N_ALL_FEATURES))
    for row_i, sid in enumerate(sids):
        for name, val in zip(rankings[sid]["features"], rankings[sid]["vals"]):
            mat[row_i, FEATURE_INDEX[name]] = val
    ranks = np.vstack([rankdata(row) for row in mat])   # average ranks; unselected features tie at the bottom

    pairs = []
    for a, b in itertools.combinations(range(len(sids)), 2):
        ra, rb = ranks[a], ranks[b]
        if np.std(ra) == 0 or np.std(rb) == 0:
            rho = float("nan")
        else:
            rho = float(np.corrcoef(ra, rb)[0, 1])   # Pearson on ranks == Spearman rho
        sid_a, sid_b = sids[a], sids[b]
        pairs.append({
            "scenario": scenario,
            "split_index_a": sid_a,
            "split_index_b": sid_b,
            "repeat_a": rankings[sid_a]["repeat"],
            "repeat_b": rankings[sid_b]["repeat"],
            "same_repeat": bool(rankings[sid_a]["repeat"] == rankings[sid_b]["repeat"]),
            "n_selected_overlap": int(len(set(rankings[sid_a]["features"]) & set(rankings[sid_b]["features"]))),
            "spearman_rho": rho,
        })
    return pairs


def build_per_repeat_rows(scenario, valid_folds, stability_repeat, gen_pairs):
    rows = []
    for repeat in range(N_REPEATS):
        fold_rows = valid_folds[valid_folds["repeat"] == repeat]
        r_vals = fold_rows["consistency_pearson_r"].to_numpy(dtype=float)
        r_finite = r_vals[np.isfinite(r_vals)]
        within = np.array([p["spearman_rho"] for p in gen_pairs
                           if p["same_repeat"] and p["repeat_a"] == repeat], dtype=float)
        within_finite = within[np.isfinite(within)]

        row = {
            "scenario": scenario,
            "repeat": repeat,
            "n_folds_valid": int(len(fold_rows)),
        }
        row.update(stability_repeat[repeat])
        row.update({
            "consistency_mean_pearson_r": float(np.mean(r_finite)) if r_finite.size else float("nan"),
            "consistency_n_finite": int(r_finite.size),
            "consistency_n_negative": int(np.sum(r_finite < 0)),
            "generalizability_within_repeat_mean_spearman_rho":
                float(np.mean(within_finite)) if within_finite.size else float("nan"),
            "generalizability_within_repeat_n_pairs": int(within_finite.size),
        })
        rows.append(row)
    return rows


def add_summary_row(rows, scenario, metric, quantity, level, values, note):
    row = {"scenario": scenario, "metric": metric, "quantity": quantity, "level": level}
    row.update(describe(values))
    row["note"] = note
    rows.append(row)


def build_summary_rows(scenario, valid_folds, per_repeat_rows, gen_pairs):
    rows = []
    per_repeat = pd.DataFrame(per_repeat_rows)

    repeat_note = "mean and sample SD (ddof=1) over the 5 repeat-level values; repeats are not independent"
    stab = [
        ("median_shap_distance_nearest_neighbor", "stability_median_shap_distance_nearest",
         "per-repeat MEDIAN over patients of the SHAP distance to the nearest same-class neighbor in the same repeat; " + repeat_note),
        ("median_shap_distance_random_neighbor", "stability_median_shap_distance_random",
         "same constraints, random same-class neighbor (fixed seed); " + repeat_note),
        ("ratio_nearest_over_random_of_medians", "stability_ratio_nearest_over_random",
         "descriptive ratio of the two medians; NOT a bounded index"),
        ("median_feature_distance_nearest_neighbor", "stability_median_feature_distance_nearest",
         "feature-space distance used to choose the nearest neighbor"),
        ("median_feature_distance_random_neighbor", "stability_median_feature_distance_random",
         "feature-space distance of the random reference pairs"),
        ("fraction_nearest_pairs_same_fold", "stability_frac_nearest_same_fold",
         "share of nearest-neighbor pairs explained by the same fold model"),
        ("fraction_random_pairs_same_fold", "stability_frac_random_same_fold",
         "share of random-reference pairs explained by the same fold model"),
    ]
    for quantity, column, note in stab:
        add_summary_row(rows, scenario, "stability", quantity, "repeat", per_repeat[column].to_numpy(), note)

    r_all = valid_folds["consistency_pearson_r"].to_numpy(dtype=float)
    add_summary_row(rows, scenario, "consistency", "pearson_r_shap_vs_pfi", "fold", r_all,
                    "taken as-is from folds CSV; negative values kept; fold values are not independent")
    add_summary_row(rows, scenario, "consistency", "mean_pearson_r_per_repeat", "repeat",
                    per_repeat["consistency_mean_pearson_r"].to_numpy(), repeat_note)

    rho_all = np.array([p["spearman_rho"] for p in gen_pairs], dtype=float)
    rho_within = np.array([p["spearman_rho"] for p in gen_pairs if p["same_repeat"]], dtype=float)
    rho_cross = np.array([p["spearman_rho"] for p in gen_pairs if not p["same_repeat"]], dtype=float)
    pair_note = "Spearman rho between fold rankings in the 49-feature space; pair values are not independent"
    add_summary_row(rows, scenario, "generalizability", "spearman_rho", "fold_pair_all", rho_all, pair_note)
    add_summary_row(rows, scenario, "generalizability", "spearman_rho", "fold_pair_within_repeat", rho_within, pair_note)
    add_summary_row(rows, scenario, "generalizability", "spearman_rho", "fold_pair_cross_repeat", rho_cross, pair_note)
    add_summary_row(rows, scenario, "generalizability", "within_repeat_mean_spearman_rho", "repeat",
                    per_repeat["generalizability_within_repeat_mean_spearman_rho"].to_numpy(), repeat_note)
    return rows


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_analysis(results_dir, out_paths):
    if N_ALL_FEATURES != 49 or COLUMN_NAMES[-1] != "Class":
        raise DataIntegrityError(
            "data_loader.COLUMN_NAMES is not '49 features + Class' (got %d features, last entry '%s')" % (
                N_ALL_FEATURES, COLUMN_NAMES[-1]))

    loaded = {}
    structures = {}
    input_info = {}
    for scenario in SCENARIOS:
        folds_path = os.path.join(results_dir, "phase3_%s_folds.csv" % scenario)
        rank_path = os.path.join(results_dir, "phase3_%s_rankings.jsonl" % scenario)
        oof_path = os.path.join(results_dir, "phase3_%s_shap_oof.jsonl" % scenario)

        valid, n_rows, n_invalid = load_valid_folds(folds_path, scenario)
        rankings, n_rank_records, n_rank_excluded = load_rankings(rank_path, scenario, valid)
        oof, n_oof_records, n_oof_excluded = load_oof(oof_path, scenario, valid, rankings)
        structures[scenario] = check_patient_structure(oof, scenario)
        loaded[scenario] = {"valid": valid, "rankings": rankings, "oof": oof}
        input_info[scenario] = {
            "folds_csv": folds_path,
            "rankings_jsonl": rank_path,
            "shap_oof_jsonl": oof_path,
            "n_csv_rows": n_rows,
            "n_valid_folds": int(len(valid)),
            "n_csv_rows_without_accuracy_excluded": n_invalid,
            "n_ranking_records": n_rank_records,
            "n_ranking_records_of_invalid_folds_excluded": n_rank_excluded,
            "n_oof_records": n_oof_records,
            "n_oof_records_of_invalid_folds_excluded": n_oof_excluded,
            "n_oof_records_used": int(len(oof)),
            "n_patients_per_repeat": int(len(structures[scenario]["patients"])),
        }
        print("[%s] loaded and validated: %d valid folds, %d out-of-fold records, %d patients per repeat" % (
            scenario, len(valid), len(oof), len(structures[scenario]["patients"])))
    check_cross_scenario(structures)

    patient_rows_all = []
    per_repeat_rows_all = []
    summary_rows_all = []
    pair_rows_all = []
    for scenario in SCENARIOS:
        data = loaded[scenario]
        patient_rows, stability_repeat = compute_stability(data["oof"], scenario)
        gen_pairs = compute_generalizability(data["rankings"], scenario)
        per_repeat_rows = build_per_repeat_rows(scenario, data["valid"], stability_repeat, gen_pairs)
        summary_rows = build_summary_rows(scenario, data["valid"], per_repeat_rows, gen_pairs)

        patient_rows_all.extend(patient_rows)
        per_repeat_rows_all.extend(per_repeat_rows)
        summary_rows_all.extend(summary_rows)
        pair_rows_all.extend(gen_pairs)

    per_repeat_df = pd.DataFrame(per_repeat_rows_all)
    summary_df = pd.DataFrame(summary_rows_all)
    patients_df = pd.DataFrame(patient_rows_all)
    pairs_df = pd.DataFrame(pair_rows_all)

    os.makedirs(results_dir, exist_ok=True)
    per_repeat_df.to_csv(out_paths["per_repeat"], index=False)
    summary_df.to_csv(out_paths["summary"], index=False)
    patients_df.to_csv(out_paths["patient_distances"], index=False)
    pairs_df.to_csv(out_paths["generalizability_pairs"], index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "compute_reliability_metrics.py",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "no_hypothesis_tests_and_no_p_values": True,
        "design": {
            "scenarios": SCENARIOS,
            "n_repeats": N_REPEATS,
            "n_splits_per_repeat": N_SPLITS,
            "valid_fold_rule": "non-null accuracy in folds CSV (status column is not used)",
            "stability": "nearest same-class neighbor inside the same repeat (49-feature global similarity "
                         "vector, Euclidean, ties -> lowest patient id), self excluded; Euclidean distance "
                         "between 49-dim SHAP vectors; per-repeat median; mean +/- sample SD (ddof=1) over "
                         "5 repeat medians; random same-class same-repeat reference with fixed seed",
            "random_reference_seed": RANDOM_REFERENCE_SEED,
            "generalizability": "Spearman rho between fold mean(|SHAP|) rankings in the 49-feature space "
                                "(0 for unselected features, average ranks for ties)",
            "consistency": "existing consistency_pearson_r per fold; negative values kept",
        },
        "inputs": input_info,
        "integrity_checks_passed": [
            "folds CSV: columns present, one valid row per split 0..24, repeat/fold consistent with split_index, scenario column correct",
            "rankings.jsonl: exactly one record per valid fold, feature names known and unique, values finite and non-negative",
            "shap_oof.jsonl: no duplicate (split, patient); n records per fold equals n_val; classes in {0,1}",
            "shap_values_full49 equals shap_values mapped through the 49-feature order for every record",
            "selected_features identical in rankings.jsonl and shap_oof.jsonl for every fold",
            "mean(|SHAP|) recomputed from shap_oof.jsonl matches rankings.jsonl for every fold",
            "each patient appears exactly once per repeat; same patient set in every repeat",
            "each patient has one class label and one similarity vector across all records",
            "both scenarios cover the same patients with the same classes and similarity vectors",
        ],
        "limitations": LIMITATIONS,
        "outputs": [os.path.basename(p) for p in out_paths.values()],
    }
    with open(out_paths["report"], "w") as handle:
        json.dump(report, handle, indent=2, default=json_default)

    print_console_summary(summary_df, out_paths)
    return report


def print_console_summary(summary_df, out_paths):
    columns = ["scenario", "metric", "quantity", "level", "n_finite", "mean", "sd_ddof1", "median", "min", "max"]
    print("\n" + "=" * 100)
    print("PHASE 3 RELIABILITY -- DESCRIPTIVE SUMMARY (no hypothesis tests, no p-values)")
    print("=" * 100)
    print(summary_df[columns].to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nLimitations (also stored in the JSON report):")
    for text in LIMITATIONS:
        print("  - " + text)
    print("\nFiles written:")
    for path in out_paths.values():
        print("  " + path)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 reliability analysis (descriptive, read-only on inputs).")
    parser.add_argument("--results-dir", default=None,
                        help="folder holding the phase3_{scenario}_* files (default: <script folder>/results)")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow overwriting existing output files (default: refuse)")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir or os.path.join(HERE, "results")
    out_paths = dict((key, os.path.join(results_dir, name)) for key, name in OUTPUT_FILES.items())

    existing = [p for p in out_paths.values() if os.path.exists(p)]
    if existing and not args.overwrite:
        print("REFUSING TO OVERWRITE existing output file(s):")
        for path in existing:
            print("  " + path)
        print("Nothing was read or written. Move or rename them, or re-run with --overwrite.")
        return 2

    try:
        run_analysis(results_dir, out_paths)
    except DataIntegrityError as exc:
        print("\nDATA INTEGRITY ERROR -- stopping, no output file was written:\n  %s" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
