"""
analyze_significance.py

Reads results/augmentation_significance.csv (produced by
run_augmentation_significance.py, one row per seed x scenario) and reports:
  1. mean +/- std of accuracy, MCC, and recall_dies per scenario
  2. pairwise Mann-Whitney U tests between every pair of scenarios, on
     both accuracy and recall_dies, at alpha = 0.05 (matching the alpha
     used elsewhere in the thesis methodology)

Usage:
    python analyze_significance.py
"""

import csv
from itertools import combinations
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

RESULTS_CSV = "results/augmentation_significance.csv"
ALPHA = 0.05


def load_results():
    by_scenario = defaultdict(lambda: defaultdict(list))
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            scenario = row["scenario"]
            for metric in ["accuracy", "mcc", "recall_dies", "recall_lives", "f1_dies"]:
                by_scenario[scenario][metric].append(float(row[metric]))
    return by_scenario


def print_summary_table(by_scenario):
    print(f"{'Scenario':<14} {'Accuracy':>18} {'MCC':>18} {'Recall(Dies)':>18}")
    for scenario, metrics in by_scenario.items():
        acc = np.array(metrics["accuracy"])
        mcc = np.array(metrics["mcc"])
        rec = np.array(metrics["recall_dies"])
        print(
            f"{scenario:<14} "
            f"{acc.mean():.3f} +/- {acc.std():.3f}   "
            f"{mcc.mean():.3f} +/- {mcc.std():.3f}   "
            f"{rec.mean():.3f} +/- {rec.std():.3f}"
        )


def pairwise_tests(by_scenario, metric):
    print(f"\n--- Pairwise Mann-Whitney U on '{metric}' (alpha={ALPHA}) ---")
    scenarios = list(by_scenario.keys())
    for a, b in combinations(scenarios, 2):
        vals_a = by_scenario[a][metric]
        vals_b = by_scenario[b][metric]
        stat, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        sig = "SIGNIFICANT" if p < ALPHA else "not significant"
        print(f"{a:<14} vs {b:<14} U={stat:.1f}  p={p:.4f}  -> {sig}")


def main():
    by_scenario = load_results()

    n_seeds = len(next(iter(by_scenario.values()))["accuracy"])
    print(f"Loaded {n_seeds} seed(s) per scenario from {RESULTS_CSV}\n")

    print("=== Summary (mean +/- std across seeds) ===")
    print_summary_table(by_scenario)

    if n_seeds < 2:
        print(
            "\nOnly 1 seed found -- Mann-Whitney U needs multiple independent "
            "samples per group to be meaningful. Run run_augmentation_significance.py "
            "with the full SEEDS list first."
        )
        return

    pairwise_tests(by_scenario, "accuracy")
    pairwise_tests(by_scenario, "recall_dies")


if __name__ == "__main__":
    main()
