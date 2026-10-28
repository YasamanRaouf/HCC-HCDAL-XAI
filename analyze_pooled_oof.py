"""
analyze_pooled_oof.py

S1 -- pooled out-of-fold comparison of the two Phase 3 scenarios
(none = no augmentation, ctgan = CTGAN augmentation of every training fold).

This script ONLY READS existing result files. It trains nothing and applies
NO hypothesis test and reports NO p-value. It uses numpy and pandas only.

WHAT IT COMPUTES
  For each scenario and each of the 5 repeats of the 5-fold CV, all 165
  patients are out-of-fold exactly once. Their stored predictions
  (`pred_proba` = P(Lives), decision rule pred_proba >= 0.5 -> Lives) are
  pooled into ONE set of 165 predictions per repeat, and these metrics are
  computed (Dies = class 0 is the "positive" class for recall/MCC):
      accuracy, MCC, recall(Dies), recall(Lives)
  Reported per scenario: mean and sample SD (ddof=1) over the 5 repeat-level
  values, plus min and max.

  None-vs-ctgan comparison (ctgan minus none), per metric:
      point estimate = difference of the two "mean over 5 repeats" values
      95% interval   = PAIRED CLUSTER BOOTSTRAP over PATIENTS: the same
                       resampled patients are used for both scenarios and
                       for all 5 repeats; the metric of each resample is the
                       mean over the repeats; percentile interval.

INPUTS (per scenario, in --results-dir, default <this folder>/results):
    phase3_{scenario}_folds.csv       valid fold = non-null `accuracy`
                                      (the `status` column is NOT used)
    phase3_{scenario}_shap_oof.jsonl  only these keys are read: split_index,
                                      repeat, fold, scenario,
                                      orig_sample_index, true_class (0 = Dies,
                                      1 = Lives), pred_proba

INTEGRITY CHECKS (the script stops and writes nothing if one fails):
    25 valid folds with unique split indices; one record per out-of-fold
    patient per fold (count equals n_val, Dies count equals n_dies_val); every
    patient exactly once per repeat; same patients and classes in every
    repeat and in both scenarios; pooled accuracy and pooled recall(Dies) of
    each repeat must reproduce, to 1e-9, the fold-size-weighted values stored
    in the folds CSV.

OUTPUTS (written into --results-dir; the script REFUSES to run if any exists,
unless --overwrite is given):
    phase3_pooled_oof_per_repeat.csv
    phase3_pooled_oof_summary.csv
    phase3_pooled_oof_comparison.csv
    phase3_pooled_oof_report.json

USAGE (hcc_thesis environment, project folder):
    python analyze_pooled_oof.py
    python analyze_pooled_oof.py --results-dir results --n-boot 10000

Exit codes: 0 ok, 1 input data not in the expected structure (nothing written),
2 refused to overwrite existing outputs.
"""

import os
import sys
import json
import argparse
import platform
import warnings
from datetime import datetime

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = ["none", "ctgan"]
N_REPEATS = 5   # must match run_phase3_scenario.py
N_SPLITS = 5    # folds per repeat; must match run_phase3_scenario.py
CLASS_DIES = 0
CLASS_LIVES = 1
THRESHOLD = 0.5
METRICS = ["accuracy", "mcc", "recall_dies", "recall_lives"]
BOOT_SEED = 42
CI_LEVEL = 0.95
DEFAULT_N_BOOT = 10000
RECONCILE_TOL = 1e-9

OUTPUT_FILES = {
    "per_repeat": "phase3_pooled_oof_per_repeat.csv",
    "summary": "phase3_pooled_oof_summary.csv",
    "comparison": "phase3_pooled_oof_comparison.csv",
    "report": "phase3_pooled_oof_report.json",
}

LIMITATIONS = [
    "The independent units are the 165 patients (63 Dies, 102 Lives). The 5 "
    "repeats re-train models on heavily overlapping data and re-predict the "
    "same patients; averaging over repeats reduces training randomness but "
    "does not add patients.",
    "The bootstrap resamples PATIENTS and keeps each repeat's trained models "
    "fixed. The interval therefore reflects patient (case-mix) sampling "
    "uncertainty only; it does not include variability from re-training on "
    "different training sets. The SD over the 5 repeat values describes that "
    "variation descriptively only (the 5 repeats are not independent).",
    "No hypothesis test and no p-value is computed. An interval that excludes "
    "zero is not, by itself, evidence of superiority: four metrics are "
    "reported, the direction of the fold-level means was seen before this "
    "analysis, and the comparison must be described as exploratory unless it "
    "was pre-specified.",
    "Pooled per-repeat metrics use all 165 out-of-fold predictions of a repeat "
    "(each from one of 5 different fold models), so they differ from the "
    "fold-level means in the folds CSV. Accuracy and recall(Dies) are "
    "reconciled against the CSV exactly; MCC and recall(Lives) are not stored "
    "there and cannot be reconciled.",
    "The decision threshold is the pipeline's own 0.5. Only the scenarios "
    "'none' and 'ctgan' exist in Phase 3 (SMOTE variants were evaluated only "
    "in Phase 2).",
]


class DataIntegrityError(Exception):
    """Inputs are not in the expected structure; stop instead of guessing."""


# ----------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------
def load_valid_folds(path, scenario):
    if not os.path.exists(path):
        raise DataIntegrityError("Missing input file: %s" % path)
    df = pd.read_csv(path)
    required = ["split_index", "repeat", "fold", "scenario", "n_val",
                "n_dies_val", "accuracy", "recall_dies"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataIntegrityError("%s: missing column(s) %s" % (path, missing))

    n_rows = int(len(df))
    valid = df[df["accuracy"].notna()].copy()
    for col in ["split_index", "repeat", "fold", "n_val", "n_dies_val"]:
        if valid[col].isna().any():
            raise DataIntegrityError("%s: valid rows have empty `%s`" % (path, col))
        valid[col] = valid[col].astype(int)

    if valid["split_index"].duplicated().any():
        dup = sorted(valid.loc[valid["split_index"].duplicated(keep=False), "split_index"].unique().tolist())
        raise DataIntegrityError("%s: split_index in more than one valid row: %s" % (path, dup))
    expected = set(range(N_REPEATS * N_SPLITS))
    found = set(valid["split_index"].tolist())
    if found != expected:
        raise DataIntegrityError("%s: expected valid folds 0..%d. Missing: %s. Unexpected: %s." % (
            path, N_REPEATS * N_SPLITS - 1, sorted(expected - found), sorted(found - expected)))
    if not (valid["repeat"] == valid["split_index"] // N_SPLITS).all():
        raise DataIntegrityError("%s: `repeat` != split_index // %d" % (path, N_SPLITS))
    if not (valid["fold"] == valid["split_index"] % N_SPLITS).all():
        raise DataIntegrityError("%s: `fold` != split_index %% %d" % (path, N_SPLITS))
    if not (valid["scenario"] == scenario).all():
        raise DataIntegrityError("%s: `scenario` column is not '%s'" % (path, scenario))
    return valid.sort_values("split_index").reset_index(drop=True), n_rows


def load_predictions(path, scenario, valid_folds):
    if not os.path.exists(path):
        raise DataIntegrityError("Missing input file: %s" % path)
    lookup = dict((int(r.split_index), r) for r in valid_folds.itertuples())
    needed = ["split_index", "repeat", "fold", "scenario", "orig_sample_index",
              "true_class", "pred_proba"]
    kept = []
    seen = set()
    n_excluded = 0
    n_lines = 0
    with open(path, "r") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise DataIntegrityError("%s line %d: invalid JSON (%s)" % (path, line_no, exc))
            missing = [k for k in needed if k not in rec]
            if missing:
                raise DataIntegrityError("%s line %d: missing key(s) %s" % (path, line_no, missing))
            if rec["scenario"] != scenario:
                raise DataIntegrityError("%s line %d: scenario '%s', expected '%s'" % (
                    path, line_no, rec["scenario"], scenario))
            sid = int(rec["split_index"])
            if sid not in lookup:
                n_excluded += 1
                continue
            row = lookup[sid]
            if int(rec["repeat"]) != int(row.repeat) or int(rec["fold"]) != int(row.fold):
                raise DataIntegrityError("%s line %d: repeat/fold disagree with the folds CSV" % (path, line_no))
            pid = int(rec["orig_sample_index"])
            if (sid, pid) in seen:
                raise DataIntegrityError("%s: patient %d appears twice in split %d" % (path, pid, sid))
            seen.add((sid, pid))
            cls = int(rec["true_class"])
            if cls not in (CLASS_DIES, CLASS_LIVES):
                raise DataIntegrityError("%s line %d: true_class=%d is neither 0 nor 1" % (path, line_no, cls))
            p = float(rec["pred_proba"])
            if not np.isfinite(p) or p < 0.0 or p > 1.0:
                raise DataIntegrityError("%s line %d: pred_proba=%r is not a probability" % (path, line_no, p))
            kept.append((sid, int(row.repeat), pid, cls, p))

    by_split = {}
    for sid, repeat, pid, cls, p in kept:
        by_split.setdefault(sid, []).append(cls)
    for sid, row in lookup.items():
        classes = by_split.get(sid, [])
        if len(classes) != int(row.n_val):
            raise DataIntegrityError("%s: split %d has %d records, folds CSV says n_val=%d" % (
                path, sid, len(classes), int(row.n_val)))
        if sum(1 for c in classes if c == CLASS_DIES) != int(row.n_dies_val):
            raise DataIntegrityError("%s: split %d Dies count differs from n_dies_val" % (path, sid))
    info = {"n_lines": n_lines, "n_used": len(kept), "n_excluded_invalid_folds": n_excluded}
    return kept, info


def build_arrays(kept, scenario):
    by_repeat = {}
    for sid, repeat, pid, cls, p in kept:
        by_repeat.setdefault(repeat, []).append((pid, cls, p))
    ref_pids = None
    ref_y = None
    prob_rows = []
    for repeat in range(N_REPEATS):
        rows = sorted(by_repeat.get(repeat, []), key=lambda t: t[0])
        pids = [t[0] for t in rows]
        y = np.array([t[1] for t in rows], dtype=int)
        if len(set(pids)) != len(pids):
            raise DataIntegrityError("%s: repeat %d contains a patient more than once" % (scenario, repeat))
        if ref_pids is None:
            ref_pids, ref_y = pids, y
        else:
            if pids != ref_pids:
                raise DataIntegrityError("%s: repeat %d does not contain the same patients as repeat 0" % (scenario, repeat))
            if not np.array_equal(y, ref_y):
                raise DataIntegrityError("%s: a patient's class differs between repeat %d and repeat 0" % (scenario, repeat))
        prob_rows.append(np.array([t[2] for t in rows], dtype=float))
    if not ref_pids:
        raise DataIntegrityError("%s: no predictions found" % scenario)
    return {"pids": ref_pids, "y": ref_y, "p": np.vstack(prob_rows)}


def reconcile_with_folds_csv(arrays, valid_folds, scenario):
    y = arrays["y"]
    pred = (arrays["p"] >= THRESHOLD).astype(int)
    dies = (y == CLASS_DIES)
    for repeat in range(N_REPEATS):
        pooled_acc = float(np.mean(pred[repeat] == y))
        pooled_rec = float(np.mean(pred[repeat][dies] == CLASS_DIES))
        rows = valid_folds[valid_folds["repeat"] == repeat]
        csv_acc = float(np.sum(rows["accuracy"] * rows["n_val"]) / np.sum(rows["n_val"]))
        csv_rec = float(np.sum(rows["recall_dies"] * rows["n_dies_val"]) / np.sum(rows["n_dies_val"]))
        if abs(pooled_acc - csv_acc) > RECONCILE_TOL or abs(pooled_rec - csv_rec) > RECONCILE_TOL:
            raise DataIntegrityError(
                "%s repeat %d: pooled accuracy/recall(Dies) from pred_proba (%.12f / %.12f) do not match the "
                "fold-size-weighted values in the folds CSV (%.12f / %.12f)" % (
                    scenario, repeat, pooled_acc, pooled_rec, csv_acc, csv_rec))


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics_from_counts(tp, fp, tn, fn):
    """Dies (class 0) is the positive class. Works on numpy arrays of any
    shape (counts of identical shape)."""
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    tn = np.asarray(tn, dtype=float)
    fn = np.asarray(fn, dtype=float)
    total = tp + fp + tn + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = (tp + tn) / total
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)   # 0 when undefined (sklearn convention)
        recall_dies = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        recall_lives = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)
    return {"accuracy": accuracy, "mcc": mcc, "recall_dies": recall_dies, "recall_lives": recall_lives}


def indicator_arrays(arrays):
    y = arrays["y"]
    pred = (arrays["p"] >= THRESHOLD).astype(int)
    dies_true = (y == CLASS_DIES)[None, :]
    dies_pred = (pred == CLASS_DIES)
    return {
        "tp": (dies_true & dies_pred).astype(float),     # shape (repeats, patients)
        "fn": (dies_true & ~dies_pred).astype(float),
        "fp": (~dies_true & dies_pred).astype(float),
        "tn": (~dies_true & ~dies_pred).astype(float),
    }


def per_repeat_metrics(ind):
    return metrics_from_counts(ind["tp"].sum(axis=1), ind["fp"].sum(axis=1),
                               ind["tn"].sum(axis=1), ind["fn"].sum(axis=1))


def bootstrap_mean_over_repeats(ind, weights):
    """weights: (B, patients) integer resampling counts. Returns, per metric,
    a (B,) array: the metric of each resample averaged over the repeats."""
    tp = weights @ ind["tp"].T     # (B, repeats)
    fp = weights @ ind["fp"].T
    tn = weights @ ind["tn"].T
    fn = weights @ ind["fn"].T
    metrics = metrics_from_counts(tp, fp, tn, fn)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, values in metrics.items():
            out[key] = np.nanmean(values, axis=1)
    return out


def percentile_interval(values, level=CI_LEVEL):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low = (1.0 - level) / 2.0 * 100.0
    high = (1.0 + level) / 2.0 * 100.0
    lo, hi = np.percentile(finite, [low, high])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_analysis(results_dir, out_paths, n_boot):
    arrays = {}
    inputs = {}
    for scenario in SCENARIOS:
        folds_path = os.path.join(results_dir, "phase3_%s_folds.csv" % scenario)
        oof_path = os.path.join(results_dir, "phase3_%s_shap_oof.jsonl" % scenario)
        valid, n_rows = load_valid_folds(folds_path, scenario)
        kept, info = load_predictions(oof_path, scenario, valid)
        arrays[scenario] = build_arrays(kept, scenario)
        reconcile_with_folds_csv(arrays[scenario], valid, scenario)
        inputs[scenario] = {"folds_csv": folds_path, "shap_oof_jsonl": oof_path,
                            "n_csv_rows": n_rows, "n_valid_folds": int(len(valid)),
                            "n_prediction_records_used": info["n_used"],
                            "n_prediction_records_of_invalid_folds_excluded": info["n_excluded_invalid_folds"],
                            "n_patients": len(arrays[scenario]["pids"])}
        print("[%s] validated: %d valid folds, %d predictions, %d patients per repeat; "
              "pooled accuracy and recall(Dies) reconcile with the folds CSV" % (
                  scenario, len(valid), info["n_used"], len(arrays[scenario]["pids"])))

    base = arrays[SCENARIOS[0]]
    other = arrays[SCENARIOS[1]]
    if base["pids"] != other["pids"] or not np.array_equal(base["y"], other["y"]):
        raise DataIntegrityError("The two scenarios do not cover the same patients with the same classes")

    n_patients = len(base["pids"])
    indicators = dict((s, indicator_arrays(arrays[s])) for s in SCENARIOS)
    point = dict((s, per_repeat_metrics(indicators[s])) for s in SCENARIOS)

    rng = np.random.default_rng(BOOT_SEED)
    weights = rng.multinomial(n_patients, np.full(n_patients, 1.0 / n_patients), size=n_boot)
    boot = dict((s, bootstrap_mean_over_repeats(indicators[s], weights)) for s in SCENARIOS)

    per_repeat_rows = []
    summary_rows = []
    y = base["y"]
    for s in SCENARIOS:
        for repeat in range(N_REPEATS):
            row = {"scenario": s, "repeat": repeat, "n_patients": n_patients,
                   "n_dies": int(np.sum(y == CLASS_DIES)), "n_lives": int(np.sum(y == CLASS_LIVES))}
            for m in METRICS:
                row[m] = float(point[s][m][repeat])
            per_repeat_rows.append(row)
        for m in METRICS:
            vals = point[s][m]
            lo, hi = percentile_interval(boot[s][m])
            summary_rows.append({
                "scenario": s, "metric": m,
                "mean_over_repeats": float(np.mean(vals)),
                "sd_ddof1_over_repeats": float(np.std(vals, ddof=1)),
                "min_over_repeats": float(np.min(vals)),
                "max_over_repeats": float(np.max(vals)),
                "boot_ci_low": lo, "boot_ci_high": hi,
                "ci_level": CI_LEVEL, "n_boot": n_boot,
            })

    comparison_rows = []
    for m in METRICS:
        mean_none = float(np.mean(point["none"][m]))
        mean_ctgan = float(np.mean(point["ctgan"][m]))
        diff_boot = boot["ctgan"][m] - boot["none"][m]
        lo, hi = percentile_interval(diff_boot)
        comparison_rows.append({
            "metric": m, "none_mean_over_repeats": mean_none, "ctgan_mean_over_repeats": mean_ctgan,
            "diff_ctgan_minus_none": mean_ctgan - mean_none,
            "boot_ci_low": lo, "boot_ci_high": hi, "ci_level": CI_LEVEL, "n_boot": n_boot,
            "note": "paired patient-level bootstrap; descriptive, no hypothesis test",
        })

    per_repeat_df = pd.DataFrame(per_repeat_rows)
    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.DataFrame(comparison_rows)

    os.makedirs(results_dir, exist_ok=True)
    per_repeat_df.to_csv(out_paths["per_repeat"], index=False)
    summary_df.to_csv(out_paths["summary"], index=False)
    comparison_df.to_csv(out_paths["comparison"], index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "analyze_pooled_oof.py",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "no_hypothesis_tests_and_no_p_values": True,
        "settings": {"scenarios": SCENARIOS, "n_repeats": N_REPEATS, "n_splits": N_SPLITS,
                     "decision_threshold": THRESHOLD, "bootstrap_seed": BOOT_SEED,
                     "n_boot": n_boot, "ci_level": CI_LEVEL,
                     "positive_class_for_recall_and_mcc": "Dies (class 0)",
                     "point_estimate": "mean over the 5 repeat-level pooled out-of-fold metrics",
                     "interval": "paired cluster (patient) bootstrap, percentile"},
        "inputs": inputs,
        "integrity_checks_passed": [
            "folds CSV: 25 valid folds (non-null accuracy), unique split indices, repeat/fold consistent, scenario correct",
            "predictions: no duplicate (split, patient); per fold the record count equals n_val and the Dies count equals n_dies_val",
            "classes in {0, 1}; pred_proba finite in [0, 1]",
            "every patient exactly once per repeat; same patients and classes in every repeat and in both scenarios",
            "pooled accuracy and pooled recall(Dies) of every repeat reproduce the fold-size-weighted CSV values to 1e-9",
        ],
        "limitations": LIMITATIONS,
        "outputs": [os.path.basename(p) for p in out_paths.values()],
    }
    with open(out_paths["report"], "w") as handle:
        json.dump(report, handle, indent=2)

    show = lambda v: "%.4f" % v
    print("\n" + "=" * 96)
    print("PHASE 3 -- POOLED OUT-OF-FOLD COMPARISON (descriptive; no hypothesis tests, no p-values)")
    print("=" * 96)
    print("\nPer scenario, mean and sample SD (ddof=1) over the 5 repeat-level values, 95% patient-bootstrap interval of the mean:")
    print(summary_df[["scenario", "metric", "mean_over_repeats", "sd_ddof1_over_repeats",
                      "min_over_repeats", "max_over_repeats", "boot_ci_low", "boot_ci_high"]]
          .to_string(index=False, float_format=show))
    print("\nctgan minus none (paired patient-level bootstrap, 95% interval):")
    print(comparison_df[["metric", "none_mean_over_repeats", "ctgan_mean_over_repeats",
                         "diff_ctgan_minus_none", "boot_ci_low", "boot_ci_high"]]
          .to_string(index=False, float_format=show))
    print("\nLimitations (also stored in the JSON report):")
    for text in LIMITATIONS:
        print("  - " + text)
    print("\nFiles written:")
    for path in out_paths.values():
        print("  " + path)


def parse_args():
    parser = argparse.ArgumentParser(description="Pooled out-of-fold comparison of none vs ctgan (read-only on inputs).")
    parser.add_argument("--results-dir", default=None,
                        help="folder with the phase3_{scenario}_* files (default: <script folder>/results)")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                        help="number of bootstrap resamples (default %d)" % DEFAULT_N_BOOT)
    parser.add_argument("--overwrite", action="store_true",
                        help="allow overwriting existing output files (default: refuse)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_boot < 1000:
        print("--n-boot must be at least 1000 for a usable percentile interval.")
        return 1
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
        run_analysis(results_dir, out_paths, args.n_boot)
    except DataIntegrityError as exc:
        print("\nDATA INTEGRITY ERROR -- stopping, no output file was written:\n  %s" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
