"""Run the GKG benchmark on the instance files published by Duguet et al.

Two configurations are run per instance.

``find``
    GZR at alpha = 1 with no warm start, stopping at the first pure equilibrium.
    This is the task solved by the branch-and-cut of Duguet et al. -- find an
    equilibrium or prove that none exists -- and is the configuration their
    reported termination rates should be compared against.

``optimal``
    RRR-BRD warm start followed by GZR optimized to gap zero, which returns the
    socially optimal equilibrium, together with the social optimum needed for
    the price of stability. Their method does not address this task; the run is
    included so that both numbers come from the same machine.

Usage::

    git clone https://github.com/AloisDuguet/branch-and-cut-for-ipgs.git
    python scripts/run_duguet_benchmark.py \\
        branch-and-cut-for-ipgs/instances/GNEP_knapsack_instances \\
        --out results/gkg_duguet_instances.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gipg.gkg.duguet_io import read_directory, check_instance
from gipg.gkg.gzr import solve_gzr
from gipg.gkg.heuristics import brd_random_restart, alpha_of_profile
from gipg.gkg.social_optimum import solve_social_optimum


def _welfare(inst, profile) -> float:
    return float((inst.p * profile).sum())


def run(indir: Path, out: Path, time_limit: float, threads: int | None,
        br_threads: int | None, skip_optimal: bool,
        limit: int | None = None) -> pd.DataFrame:
    rows = []
    t_start = time.time()

    for k, inst in enumerate(read_directory(indir), start=1):
        if limit is not None and k > limit:
            break
        complaints = check_instance(inst)
        if complaints:
            print(f"  ! {inst.meta['file']}: {'; '.join(complaints)}", file=sys.stderr)

        row = {
            "file": inst.meta["file"],
            "n": inst.n,
            "m": inst.m,
            "cap_factor": inst.cap_factor,
            "corr": inst.corr,
            "replication": inst.meta["replication"],
            "threads": threads,
            "br_threads": br_threads,
        }

        # --- configuration "find": no warm start, stop at the first PNE -----
        res = solve_gzr(
            inst, alpha=1.0, time_limit=time_limit, threads=threads,
            br_threads=br_threads, warm_start=None, stop_at_first=True,
            verbose=False,
        )
        found = res.status in ("FEASIBLE", "OPTIMAL") and res.profile is not None
        row.update(
            find_status=res.status,
            find_found=found,
            find_time=res.runtime,
            find_cuts=res.cuts_added,
            find_br_calls=res.br_calls,
            find_first_pne_time=res.first_pne_time,
        )

        # --- configuration "optimal": warm start, optimize over the PNE set --
        if not skip_optimal:
            x_brd, brd_pne, brd_time = brd_random_restart(
                inst, max_init=3, max_round=15, seed=0,
            )
            row.update(
                brd_found_pne=bool(brd_pne),
                brd_time=brd_time,
                brd_alpha=alpha_of_profile(inst, x_brd) if brd_pne else float("inf"),
            )
            opt = solve_gzr(
                inst, alpha=1.0, time_limit=max(time_limit - brd_time, 60.0),
                threads=threads, br_threads=br_threads,
                warm_start=x_brd if brd_pne else None,
                stop_at_first=False, verbose=False,
            )
            row.update(
                opt_status=opt.status,
                opt_time=brd_time + opt.runtime,
                opt_cuts=opt.cuts_added,
                opt_mip_gap=opt.mip_gap,
                best_pne_welfare=_welfare(inst, opt.profile) if opt.profile is not None else None,
            )

            so = solve_social_optimum(inst, time_limit=time_limit, threads=threads)
            row.update(so_status=so.status, so_time=so.runtime, so_welfare=so.opt_cost)
            if row["best_pne_welfare"] and so.opt_cost:
                row["pos"] = so.opt_cost / row["best_pne_welfare"]

        rows.append(row)
        if k % 20 == 0 or k == 1:
            print(f"  {k} instances, {time.time() - t_start:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def report(df: pd.DataFrame) -> None:
    n_inst = len(df)
    print(f"\nMaster threads {df['threads'].iloc[0]}, "
          f"best-response threads {df['br_threads'].iloc[0]}")
    found = int(df["find_found"].sum())
    print(f"\nInstances: {n_inst}")
    print(f"PNE found (no warm start, first PNE): {found} ({found / n_inst:.1%})")
    print(f"Mean time {df['find_time'].mean():.2f}s, max {df['find_time'].max():.1f}s, "
          f"mean CEIs {df['find_cuts'].mean():.1f}")

    print("\nSolved out of 60 per (n, m) cell:")
    grid = df.pivot_table(index="n", columns="m", values="find_found", aggfunc="sum")
    print(grid.astype(int).to_string())
    print("\nMean seconds per (n, m) cell:")
    print(df.pivot_table(index="m", columns="n", values="find_time",
                         aggfunc="mean").round(2).to_string())

    if "opt_status" in df:
        n_opt = int((df["opt_status"] == "OPTIMAL").sum())
        print(f"\nSocially optimal PNE proved: {n_opt} of {n_inst}, "
              f"mean {df['opt_time'].mean():.2f}s (BRD warm start included)")
        if "pos" in df:
            print(f"Mean price of stability {df['pos'].mean():.4f}, "
                  f"social optimum mean {df['so_time'].mean():.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("indir", type=Path,
                    help="directory of GNEP_knapsack_instances/*.txt")
    ap.add_argument("--out", type=Path,
                    default=Path("results/gkg_duguet_instances.csv"))
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=16,
                    help="threads for the master problem and social optimum")
    ap.add_argument("--br-threads", type=int, default=None,
                    help="threads for best-response subproblems "
                         "(default: same as --threads)")
    ap.add_argument("--skip-optimal", action="store_true",
                    help="run only the equilibrium-finding configuration")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many instances (for a smoke test)")
    args = ap.parse_args()

    if not args.indir.is_dir():
        sys.exit(f"not a directory: {args.indir}")
    n_files = len(list(args.indir.glob('*.txt')))
    if n_files == 0:
        sys.exit(f"no .txt instance files in {args.indir}")
    br_threads = args.threads if args.br_threads is None else args.br_threads
    print(f"Reading {n_files} instance files from {args.indir}")
    print(f"Master threads {args.threads}, best-response threads {br_threads}, "
          f"time limit {args.time_limit:.0f}s")
    if not args.skip_optimal and args.threads == 1:
        print("  note: RRR-BRD solves its best responses with 16 threads; "
              "use --skip-optimal for a strictly single-threaded run",
              file=sys.stderr)

    df = run(args.indir, args.out, args.time_limit, args.threads,
             br_threads, args.skip_optimal, args.limit)
    report(df)
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
