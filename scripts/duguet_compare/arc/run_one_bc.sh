#!/usr/bin/env bash
# Run Duguet et al.'s B&C on one instance; resume-safe; keeps their log and one-line result.
#   run_one_bc.sh <main-binary> <instance-path> <time_limit_s> <out-dir>
set -uo pipefail
BIN="$1"; INST="$2"; TL="$3"; OUT="$4"; f="$(basename "$INST" .txt)"
mkdir -p "$OUT"
[ -s "$OUT/$f.result" ] && exit 0
cd "$(dirname "$BIN")"
"$BIN" "$INST" GNEP-fullInteger basicAlgorithm intersectionCuts 0 0 manyEtasGurobi mostFractional "$TL" 1 "$OUT/$f.result.tmp" > "$OUT/$f.log" 2>&1
if [ -s "$OUT/$f.result.tmp" ]; then mv "$OUT/$f.result.tmp" "$OUT/$f.result"; else echo "NO_RESULT $f" >> "$OUT/_failures.txt"; fi
