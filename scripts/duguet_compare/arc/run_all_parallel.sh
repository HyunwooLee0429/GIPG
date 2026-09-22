#!/usr/bin/env bash
# Full Duguet-vs-GZR run on one node, in parallel.  Run inside an exclusive allocation:
#   salloc --nodes=1 --exclusive --time=1-00:00:00 --account=<alloc>   (then ssh/srun to the node)
#   bash scripts/duguet_compare/arc/run_all_parallel.sh <path-to-branch-and-cut-for-ipgs> [time_limit=600] [jobs=32]
# Resume-safe: rerunning skips finished instances.  Output under scripts/duguet_compare/full/.
set -euo pipefail
REPO="${1:?}"; TL="${2:-600}"; JOBS="${3:-32}"
HERE="$(cd "$(dirname "$0")" && pwd)"; DC="$HERE/.."; GIPG="$(cd "$DC/../.." && pwd)"
BIN="$REPO/code/branch-and-prune-and-cut/build/main"; INST="$REPO/instances/GNEP_knapsack_instances"
FULL="$DC/full"; mkdir -p "$FULL/bc" "$FULL/gzr_logs"
VENV="$HOME/venvs/gipg"; export PATH="$VENV/bin:$PATH"
ls "$INST"/*.txt | xargs -n1 basename | sort > "$FULL/all_instances.txt"
echo "$(wc -l < "$FULL/all_instances.txt") instances, TL=${TL}s, ${JOBS} parallel B&C jobs (each single-threaded by their ThreadLimit=1)"
echo "host $(hostname), $(nproc) cores, started $(date -Is)" >> "$FULL/run_log.txt"

# --- their B&C, JOBS at a time (their code caps Gurobi at 1 thread)
sed "s|^|$INST/|" "$FULL/all_instances.txt" | xargs -P "$JOBS" -I{} bash "$HERE/run_one_bc.sh" "$BIN" {} "$TL" "$FULL/bc"
python3 "$HERE/collect_bc.py" "$FULL/bc" "$FULL/results_duguet_bc.csv"

# --- ours, single-threaded to match, 4 processes over disjoint quarters of the list
split -n l/4 -d "$FULL/all_instances.txt" "$FULL/gzr_part_"
for part in "$FULL"/gzr_part_0?; do
  python3 "$DC/run_gzr_on_duguet.py" --instances-dir "$INST" --list "$part" --out "$part.csv" \
      --time-limit "$TL" --threads 1 --log-dir "$FULL/gzr_logs" > "$part.stdout" 2>&1 &
done; wait
python3 - "$FULL" <<'PY'
import sys, glob, pandas as pd, os
F = sys.argv[1]
g = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(F, "gzr_part_0?.csv")))])
g.to_csv(os.path.join(F, "results_gzr.csv"), index=False)
b = pd.read_csv(os.path.join(F, "results_duguet_bc.csv"))
m = g.merge(b, on="file", how="outer"); m.to_csv(os.path.join(F, "comparison_full.csv"), index=False)
m["n"] = m.file.str.split("-").str[0].astype(int); m["m"] = m.file.str.split("-").str[1].astype(int)
print(m.groupby(["n","m"]).agg(N=("file","size"), bc_solved=("their_status", lambda s:(s=="SOLUTION_FOUND").sum()),
      bc_time=("their_time","mean"), bc_cuts=("their_cuts","mean"),
      gzr_first=("first_status", lambda s:(s=="FEASIBLE").sum()), gzr_first_time=("first_time","mean"),
      gzr_opt=("gzr_status", lambda s:(s=="OPTIMAL").sum()), gzr_time=("gzr_total_time","mean")).round(2).to_string())
PY
echo "finished $(date -Is)" >> "$FULL/run_log.txt"
