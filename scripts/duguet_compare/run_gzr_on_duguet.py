"""Run GZR (ours) on Duguet et al. GNEP knapsack instance files.

Usage:
  python run_gzr_on_duguet.py --instances-dir <...>/instances/GNEP_knapsack_instances \
      --list sample.txt --out results_gzr.csv [--time-limit 600] [--threads 1] [--log-dir logs/gzr]

For each instance, records three runs (mirrors notebooks/gkg_run_benchmark.ipynb):
  (a) RRR-BRD warm start + GZR optimizing social welfare (paper's Table 1 setting),
  (b) GZR stop_at_first, no warm start  -- the like-for-like counterpart of
      Duguet et al.'s branch-and-cut, which stops at the first NE found,
  (c) social optimum reference MILP (no CEIs).

With --log-dir, every instance gets a folder <log-dir>/<instance>/ containing the
Gurobi logs of the three models (gzr_master.log, first_master.log, so.log), the
CEI trace of run (a) (cuts.jsonl: time, player, best-response profit, regret),
and provenance.json (git commit, Gurobi version, host, threads, timestamps).
Resume-safe: instances already present in --out are skipped.
"""
from __future__ import annotations
import argparse, csv, json, os, platform, socket, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from parse_duguet import parse_duguet_gkg
from gipg.gkg.heuristics import brd_random_restart, alpha_of_profile
from gipg.gkg.gzr import solve_gzr
from gipg.gkg.social_optimum import solve_social_optimum, compute_pos

FIELDS = ["file", "n", "m", "cf", "corr",
          "brd_found_pne", "brd_time",
          "gzr_status", "gzr_time", "gzr_total_time", "gzr_cuts", "gzr_nodes", "gzr_first_pne_time", "gzr_welfare", "gzr_gap",
          "first_status", "first_time", "first_cuts", "first_nodes", "first_welfare",
          "so_status", "so_time", "so_welfare", "pos", "threads", "host"]


def _git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_one(path, time_limit, threads, log_dir):
    inst = parse_duguet_gkg(path)
    name = os.path.basename(path).replace(".txt", "")
    ld = None
    if log_dir:
        ld = os.path.join(log_dir, name); os.makedirs(ld, exist_ok=True)
    lf = (lambda k: os.path.join(ld, k)) if ld else (lambda k: None)
    row = {"file": os.path.basename(path), "n": inst.n, "m": inst.m, "cf": inst.cap_factor, "corr": inst.corr,
           "threads": threads, "host": socket.gethostname()}
    t_start = time.time()
    x_brd, brd_pne, brd_time = brd_random_restart(inst, max_init=3, max_round=15, seed=0)
    row["brd_found_pne"] = brd_pne; row["brd_time"] = round(brd_time, 3)
    warm = x_brd if brd_pne else None
    r = solve_gzr(inst, alpha=1.0, time_limit=max(time_limit - brd_time, 10.0), warm_start=warm,
                  stop_at_first=False, threads=threads, br_threads=threads,
                  log_file=lf("gzr_master.log"), trace_cuts=bool(ld))
    row.update(gzr_status=r.status, gzr_time=round(r.runtime, 3), gzr_total_time=round(brd_time + r.runtime, 3),
               gzr_cuts=r.cuts_added, gzr_nodes=r.node_count, gzr_first_pne_time=r.first_pne_time,
               gzr_welfare=r.obj_val, gzr_gap=r.mip_gap)
    r1 = solve_gzr(inst, alpha=1.0, time_limit=time_limit, warm_start=None, stop_at_first=True,
                   threads=threads, br_threads=threads, log_file=lf("first_master.log"))
    row.update(first_status=r1.status, first_time=round(r1.runtime, 3), first_cuts=r1.cuts_added,
               first_nodes=r1.node_count, first_welfare=r1.obj_val)
    so = solve_social_optimum(inst, time_limit=time_limit, threads=threads, log_file=lf("so.log"))
    row.update(so_status=so.status, so_time=round(so.runtime, 3), so_welfare=so.opt_cost)
    row["pos"] = compute_pos(so.opt_cost, r.obj_val) if (r.obj_val is not None and so.opt_cost) else None
    if ld:
        with open(os.path.join(ld, "cuts.jsonl"), "w") as f:
            for rec in (r.cut_trace or []):
                f.write(json.dumps(rec) + "\n")
        import gurobipy as gp
        prov = {"instance": os.path.basename(path), "gipg_commit": _git_commit(os.path.join(os.path.dirname(__file__), "..", "..")),
                "gurobi_version": ".".join(map(str, gp.gurobi.version())), "python": platform.python_version(),
                "host": socket.gethostname(), "platform": platform.platform(), "threads": threads,
                "time_limit": time_limit, "brd": {"max_init": 3, "max_round": 15, "seed": 0},
                "started": t_start, "finished": time.time()}
        json.dump(prov, open(os.path.join(ld, "provenance.json"), "w"), indent=1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-dir", required=True)
    ap.add_argument("--list", required=True, help="text file, one instance basename per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=None, help="Gurobi threads (default: package default, 16)")
    ap.add_argument("--log-dir", default=None)
    a = ap.parse_args()
    names = [l.strip() for l in open(a.list) if l.strip() and not l.startswith("#")]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {r["file"] for r in csv.DictReader(f)}
    mode = "a" if done else "w"
    with open(a.out, mode, newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w": wr.writeheader()
        for nm in names:
            if nm in done:
                continue
            row = run_one(os.path.join(a.instances_dir, nm), a.time_limit, a.threads, a.log_dir)
            wr.writerow(row); f.flush()
            print(f"{nm:28s} GZR {row['gzr_status']:10s} {row['gzr_total_time']:7.2f}s  first-PNE {row['first_status']:10s} {row['first_time']:7.2f}s  cuts {row['first_cuts']}", flush=True)


if __name__ == "__main__":
    main()
