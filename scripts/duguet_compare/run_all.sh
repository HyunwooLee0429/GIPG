#!/usr/bin/env bash
# One-shot sample comparison (~10 min). Run from anywhere:
#   bash scripts/duguet_compare/run_all.sh <path-to-branch-and-cut-for-ipgs> [time_limit_s=40]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="${1:?}"; TL="${2:-40}"
cd "$HERE/../.." && pip install -q -e . >/dev/null
python3 "$HERE/run_gzr_on_duguet.py" --instances-dir "$REPO/instances/GNEP_knapsack_instances" \
  --list "$HERE/sample/sample.txt" --out "$HERE/sample/results_gzr.csv" --time-limit "$TL"
bash "$HERE/run_duguet_bc.sh" "$REPO" "$TL"
python3 "$HERE/compare.py"
