"""Collect Duguet B&C one-line results into a CSV.  python collect_bc.py <out-dir> <csv>"""
import sys, glob, os, csv
out, dst = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(out, "*.result"))):
    t = open(p).read().strip().splitlines()[-1].split(",")
    # trailing fields: STATUS, NE_FOUND, NODES, CUTS, TIME, CUT_TIME, ANTICYCLING
    rows.append({"file": os.path.basename(p)[:-7] + ".txt", "their_status": t[-7], "their_ne_found": t[-6],
                 "their_nodes": t[-5], "their_cuts": t[-4], "their_time": t[-3], "their_cut_time": t[-2]})
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} results -> {dst}")
