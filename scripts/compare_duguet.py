"""Build the per-cell comparison against Duguet et al. from the two result files.

Their raw results are published in their own repository and are not redistributed
here; clone it and point ``--theirs`` at the file::

    git clone https://github.com/AloisDuguet/branch-and-cut-for-ipgs.git
    python scripts/compare_duguet.py \\
        --ours results/gkg_duguet_1thread.csv \\
        --theirs branch-and-cut-for-ipgs/results/GNEPKnapsack/results.csv

Their file carries one header name fewer than it has data fields, so the columns
are read from the end, where the order is documented as ``SOLVE_STATUS``,
``NUMBER_OF_NE_FOUND``, ``NODE_EXPLORED``, ``CUTS_ADDED``, ``TIME``,
``ANTICYCLING_MEASURES_TAKEN``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MS = [5, 10, 15, 20, 30, 40, 50]
NS = [2, 3, 4]


def read_theirs(path: Path) -> pd.DataFrame:
    lines = [ln for ln in path.read_text().splitlines()[1:] if ln.strip()]
    rows = [ln.split(",") for ln in lines]
    df = pd.DataFrame({
        "file": [r[1].split("/")[-1] for r in rows],
        "status": [r[-6] for r in rows],
        "nodes": [float(r[-4]) for r in rows],
        "cuts": [float(r[-3]) for r in rows],
        "time": [float(r[-2]) for r in rows],
    })
    df["solved"] = df["status"] != "TIME_LIMIT_REACHED"
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=Path, default=Path("results/gkg_duguet_1thread.csv"))
    ap.add_argument("--theirs", type=Path, required=True)
    args = ap.parse_args()

    theirs = read_theirs(args.theirs)
    ours = pd.read_csv(args.ours)
    df = theirs.merge(ours[["file", "n", "m", "find_found", "find_time", "find_cuts"]],
                      on="file", validate="one_to_one")
    if len(df) != len(theirs):
        raise SystemExit(f"matched {len(df)} of {len(theirs)} instances")

    print(f"Instances {len(df)}")
    print(f"  Duguet et al.  solved {int(df.solved.sum()):5d}  "
          f"mean {df.time.mean():8.1f}s  mean cuts {df.cuts.mean():9.0f}  "
          f"max cuts {int(df.cuts.max())}")
    print(f"  CEI-GZR        solved {int(df.find_found.sum()):5d}  "
          f"mean {df.find_time.mean():8.3f}s  mean cuts {df.find_cuts.mean():9.1f}  "
          f"max cuts {int(df.find_cuts.max())}")

    both = df[df.solved]
    print(f"\nOn the {len(both)} instances both resolve, median time ratio "
          f"{(both.time / both.find_time).median():.0f}")
    print(f"Median cut-count ratio over all instances "
          f"{(df.cuts / df.find_cuts).median():.0f}")

    print("\nTable rows: m, then for each n: their solved, their mean cuts, "
          "our mean time, our mean CEIs")
    for m in MS:
        cells = []
        for n in NS:
            g = df[(df.n == n) & (df.m == m)]
            cells.append(f"{int(g.solved.sum())} & {g.cuts.mean():,.0f} & "
                         f"{g.find_time.mean():.2f} & {g.find_cuts.mean():.1f}")
        print(f"{m} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
