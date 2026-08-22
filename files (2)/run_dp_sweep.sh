#!/bin/bash
# run_dp_sweep.sh
#
# Runs the federated HCC simulation once per noise level, all appending to
# the same results/fl_metrics.csv, so you end up with a privacy-utility
# comparison table in one file. Takes a few minutes total.
#
# Usage:
#   bash run_dp_sweep.sh

set -e  # stop immediately if any run errors out

echo "=== Run 1/4: no-DP control (noise_multiplier=0.0) ==="
DP_ENABLED=0 python run_federated_hcc.py

echo "=== Run 2/4: noise_multiplier=0.1 ==="
DP_ENABLED=1 NOISE_MULTIPLIER=0.1 python run_federated_hcc.py

echo "=== Run 3/4: noise_multiplier=0.3 ==="
DP_ENABLED=1 NOISE_MULTIPLIER=0.3 python run_federated_hcc.py

echo "=== Run 4/4: noise_multiplier=0.5 ==="
DP_ENABLED=1 NOISE_MULTIPLIER=0.5 python run_federated_hcc.py

echo ""
echo "All 4 runs complete. Results are in results/fl_metrics.csv"
