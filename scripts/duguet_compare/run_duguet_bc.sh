#!/usr/bin/env bash
# Build Duguet et al.'s branch-and-cut (C++) and run it on the sample instances.
# Usage: ./run_duguet_bc.sh <path-to-branch-and-cut-for-ipgs> [time_limit_s=40] [list=sample/sample.txt]
# Needs: cmake >= 3.26, a C++20 compiler, Eigen3, Gurobi with the C++ library, GUROBI_HOME set
#        (e.g. export GUROBI_HOME=/Library/gurobi1300/macos_universal2).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${1:?path to branch-and-cut-for-ipgs}"; TL="${2:-40}"; LIST="${3:-$HERE/sample/sample.txt}"
CODE="$REPO/code/branch-and-prune-and-cut"; INST="$REPO/instances/GNEP_knapsack_instances"
OUT="$HERE/sample/results_duguet_bc.csv"

if [ -z "${GUROBI_HOME:-}" ]; then
  G=$(ls -d /Library/gurobi*/macos_universal2 /opt/gurobi*/linux64 2>/dev/null | sort | tail -1 || true)
  [ -n "$G" ] && export GUROBI_HOME="$G"
fi
echo "GUROBI_HOME=${GUROBI_HOME:-<unset>}"
# On macOS the shipped libgurobi_c++.a is built with clang; make sure we use clang++.
CXX_FLAG=""; command -v clang++ >/dev/null && CXX_FLAG="-DCMAKE_CXX_COMPILER=$(command -v clang++)"
if [ ! -x "$CODE/build/main" ]; then
  cmake -DCMAKE_BUILD_TYPE=Release $CXX_FLAG -S "$CODE" -B "$CODE/build"
  cmake --build "$CODE/build" --target main -j
fi

echo "file,their_status,their_ne_found,their_nodes,their_cuts,their_time" > "$OUT"
while read -r f; do
  [ -z "$f" ] && continue; case "$f" in \#*) continue;; esac
  RES="$HERE/sample/_tmp_result.txt"; rm -f "$RES"
  ( cd "$CODE/build" && ./main "$INST/$f" GNEP-fullInteger basicAlgorithm intersectionCuts 0 0 manyEtasGurobi mostFractional "$TL" 0 "$RES" > "$HERE/sample/_log_$f.log" 2>&1 ) || true
  # result line: result,<instance>,GNEP,GNEP-fullInteger,...,FEASIBILITY_TOL,STATUS,NE_FOUND,NODES,CUTS,TIME,CUT_TIME,ANTICYCLING
  if [ -f "$RES" ]; then
    LINE=$(tail -1 "$RES")
    python3 - "$f" "$LINE" >> "$OUT" <<'PY'
import sys; f, line = sys.argv[1], sys.argv[2]; t = line.split(',')
# columns from the end: [-7]=STATUS [-6]=NE_FOUND [-5]=NODES [-4]=CUTS [-3]=TIME [-2]=CUT_TIME [-1]=ANTICYCLING
print(",".join([f, t[-7], t[-6], t[-5], t[-4], t[-3]]))
PY
  else
    echo "$f,NO_RESULT,,,," >> "$OUT"
  fi
  tail -1 "$OUT"
done < "$LIST"
rm -f "$HERE/sample/_tmp_result.txt"
echo "wrote $OUT"
