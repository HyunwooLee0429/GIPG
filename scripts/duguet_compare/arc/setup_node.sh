#!/usr/bin/env bash
# One-time setup on an ARC (TinkerCliffs) node or login node.
#   bash scripts/duguet_compare/arc/setup_node.sh <path-to-branch-and-cut-for-ipgs>
# Builds Duguet et al.'s C++ code with GCC, creates a venv with gurobipy + gipg,
# and checks the Gurobi license on this host. Run the build part on the LOGIN
# node if compute nodes have no internet (their CMake fetches Catch2 from GitHub).
set -euo pipefail
REPO="${1:?path to branch-and-cut-for-ipgs clone}"
HERE="$(cd "$(dirname "$0")" && pwd)"; GIPG="$(cd "$HERE/../../.." && pwd)"
CODE="$REPO/code/branch-and-prune-and-cut"

module reset >/dev/null 2>&1 || true
module load GCC/13.2.0 CMake/3.27.6-GCCcore-13.2.0 Eigen/3.4.0-GCCcore-13.2.0 Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
 || { echo "module load failed; run 'module spider GCC CMake Eigen Python' and edit this line"; exit 1; }

# --- their code (upstream, unpatched: GCC has bits/stdc++.h and current_zone)
if [ ! -x "$CODE/build/main" ]; then
  export EIGEN_PATH="${EBROOTEIGEN:-}/include"
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$(command -v g++)" -S "$CODE" -B "$CODE/build"
  cmake --build "$CODE/build" --target main -j 8
fi
echo "built: $CODE/build/main"

# --- ours
VENV="$HOME/venvs/gipg"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "gurobipy>=12,<14" numpy pandas
"$VENV/bin/pip" install -q -e "$GIPG"

# --- license check on THIS host (named-user license is bound to the hostname)
echo "host: $(hostname)"
"$VENV/bin/python" - <<'PY'
import gurobipy as gp
m = gp.Model(); x = m.addVars(3000); m.setObjective(x.sum()); m.addConstr(x.sum() <= 1); m.optimize()
print("Gurobi", ".".join(map(str, gp.gurobi.version())), "full license OK (3000-var model solved)")
PY
