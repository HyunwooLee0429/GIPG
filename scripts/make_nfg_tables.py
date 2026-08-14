#!/usr/bin/env python3
"""Generate all NFG LaTeX tables from a single pair of result CSVs.

Both the VEST-on and VEST-off runs are read once, and every NFG table in the
paper is derived from that single pair.  This guarantees the four tables are
mutually consistent -- the failure mode that previously produced a 66.1 s entry
in tab:vest-comparison against a 62.0 s entry in tab:nfg-summary.

Usage
-----
    python make_nfg_tables.py                       # default paths
    python make_nfg_tables.py --on A.csv --off B.csv --out tables_nfg.tex

Notes
-----
* POS is averaged over every instance for which GZR returned a PNE.  Where the
  social-optimum run hit its time limit its incumbent is an upper bound on the
  optimal social cost, so that row's ratio is a lower bound on the true price of
  stability rather than an invalid number; such rows are counted on stderr.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_ON = HERE.parent / "results" / "nfg_exp1.csv"
DEFAULT_OFF = HERE.parent / "results" / "nfg_exp1_no_vest.csv"

CAP_LABEL = {"loose": "Loose", "tight": "Tight"}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    # de-duplicate repeated column names (some exports duplicate n_players)
    seen: dict[str, int] = {}
    cols = []
    for c in d.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    d.columns = cols
    for c in ("brd_time", "gzr_time", "gzr_cuts", "pos", "so_cost", "best_pne_cost", "brd_alpha"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def pos_series(d: pd.DataFrame) -> pd.Series:
    """POS over all rows.

    A social-optimum run that hits its time limit returns a feasible incumbent,
    which is an upper bound on the true optimal social cost.  The resulting
    ratio ``best_pne_cost / so_cost`` is therefore a *lower* bound on the true
    price of stability -- conservative, and still at least one.  Such rows are
    kept; the diagnostics below report how many there are.
    """
    return d["pos"]


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def agg(g: pd.DataFrame) -> dict:
    st = g["gzr_status"]
    p = pos_series(g).dropna()
    return {
        "count": len(g),
        "brd": int(g["brd_found_pne"].sum()) if "brd_found_pne" in g else len(g),
        "gzr": int((~st.isin(["INFEASIBLE"])).sum()),
        "alpha": g["brd_alpha"].mean() if "brd_alpha" in g else np.nan,
        "t_brd": g["brd_time"].mean(),
        "t_gzr": g["gzr_time"].mean(),
        "pos": p.mean() if len(p) else np.nan,
        "pos_n": len(p),
        "cei": g["gzr_cuts"].mean() if "gzr_cuts" in g else np.nan,
        "opt": int((st == "OPTIMAL").sum()),
        "inf": int((st == "INFEASIBLE").sum()),
        "tl": int((st == "TIME_LIMIT").sum()),
    }


def fmt(v, nd=1):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def row(a: dict, lead: str) -> str:
    return (
        f"{lead} & {a['count']} & {a['brd']} & {a['gzr']} & {fmt(a['alpha'],3)} & "
        f"{fmt(a['t_brd'])} & {fmt(a['t_gzr'])} & {fmt(a['pos'],3)} & {fmt(a['cei'])} & "
        f"{a['opt']} & {a['inf']} & {a['tl']} \\\\"
    )


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def t_summary(d: pd.DataFrame) -> str:
    L = [
        r"\begin{table}[h!]",
        r"\caption{NFG results by capacity level.}",
        r"\label{tab:nfg-summary}",
        r"\centering", r"\small",
        r"\begin{tabular}{lccccccccccc}", r"\toprule",
        r"Capacity & Count & BRD & GZR & $\alpha_{\mathrm{brd}}$ & $T_{\mathrm{BRD}}$ & "
        r"$T_{\mathrm{GZR}}$ & POS & $\#_{\mathrm{CEI}}$ & OPT & INF & TL \\ \midrule",
    ]
    for cap in ("loose", "tight"):
        g = d[d["capacity_level"] == cap]
        if len(g):
            L.append(row(agg(g), CAP_LABEL[cap]))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def t_by_cap_n(d: pd.DataFrame, npc: str) -> str:
    L = [
        r"\begin{table}[h!]",
        r"\caption{NFG results by capacity level and number of players.}",
        r"\label{tab:nfg-by-cap-n}",
        r"\centering", r"\small",
        r"\begin{tabular}{cccccccccccccc}", r"\toprule",
        r"Cap.\ & $n$ & Count & BRD & GZR & $\alpha_{\mathrm{brd}}$ & $T_{\mathrm{BRD}}$ & "
        r"$T_{\mathrm{GZR}}$ & POS & $\#_{\mathrm{CEI}}$ & OPT & INF & TL \\ \midrule",
    ]
    for cap in ("loose", "tight"):
        for n in sorted(d.loc[d["capacity_level"] == cap, npc].unique()):
            g = d[(d["capacity_level"] == cap) & (d[npc] == n)]
            L.append(row(agg(g), f"{CAP_LABEL[cap]} & {n}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def t_comp(d: pd.DataFrame, npc: str) -> str:
    L = [
        r"\begin{table}[h!]",
        r"\caption{Comprehensive NFG results by capacity, number of players, and layers.}",
        r"\label{tab:nfg-comp}",
        r"\centering", r"\small",
        r"\begin{tabular}{cccccccccccccc}", r"\toprule",
        r"Cap.\ & $n$ & Layers & Count & BRD & GZR & $\alpha_{\mathrm{brd}}$ & "
        r"$T_{\mathrm{BRD}}$ & $T_{\mathrm{GZR}}$ & POS & $\#_{\mathrm{CEI}}$ & "
        r"OPT & INF & TL \\ \midrule",
    ]
    for cap in ("loose", "tight"):
        for n in sorted(d.loc[d["capacity_level"] == cap, npc].unique()):
            for L_ in sorted(d.loc[(d["capacity_level"] == cap) & (d[npc] == n), "n_layers"].unique()):
                g = d[(d["capacity_level"] == cap) & (d[npc] == n) & (d["n_layers"] == L_)]
                L.append(row(agg(g), f"{CAP_LABEL[cap]} & {n} & {L_}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def t_vest(on: pd.DataFrame, off: pd.DataFrame) -> str:
    """NFG block of tab:vest-comparison (ISBP blocks are produced elsewhere)."""
    a_on, a_off = agg(on), agg(off)

    key = "tag" if "tag" in on.columns and "tag" in off.columns else None
    per = np.nan
    if key:
        m = on.merge(off, on=key, suffixes=("_on", "_off"))
        ok = m[(m["gzr_status_on"] == "OPTIMAL") & (m["gzr_status_off"] == "OPTIMAL")]
        if len(ok):
            per = (ok["gzr_time_off"] / ok["gzr_time_on"].replace(0, np.nan)).mean()
    aggr = a_off["t_gzr"] / a_on["t_gzr"] if a_on["t_gzr"] else np.nan

    return "\n".join([
        r"% --- NFG block for tab:vest-comparison ---",
        r"\multirow{2}{*}{NFG}",
        f"  & wVEST on  & {a_on['count']} & {fmt(a_on['t_gzr'])} & {fmt(a_on['cei'])} & "
        f"{a_on['opt']} & {a_on['inf']} & {a_on['tl']} & "
        rf"\multirow{{2}}{{*}}{{{fmt(aggr,3)}}} & \multirow{{2}}{{*}}{{{fmt(per,3)}}} \\",
        f"  & wVEST off & {a_off['count']} & {fmt(a_off['t_gzr'])} & {fmt(a_off['cei'])} & "
        f"{a_off['opt']} & {a_off['inf']} & {a_off['tl']} & & \\\\",
    ])


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", type=Path, default=DEFAULT_ON)
    ap.add_argument("--off", type=Path, default=DEFAULT_OFF)
    ap.add_argument("--out", type=Path, default=HERE.parent / "results" / "nfg_tables.tex")
    args = ap.parse_args()

    for p in (args.on, args.off):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    on, off = load(args.on), load(args.off)
    npc = "n_players" if "n_players" in on.columns else "n_players.1"

    # ---- consistency / sanity report -------------------------------------
    w = sys.stderr
    print("=" * 66, file=w)
    print(f"VEST on : {len(on):4d} rows   mean T_GZR = {on['gzr_time'].mean():7.2f} s", file=w)
    print(f"VEST off: {len(off):4d} rows   mean T_GZR = {off['gzr_time'].mean():7.2f} s", file=w)

    for tag, d in (("on", on), ("off", off)):
        if "so_status" in d.columns:
            nc = int((d["so_status"] != "OPTIMAL").sum())
            if nc:
                print(f"  [{tag}] {nc} instance(s) with a time-limited SO reference "
                      f"-> their POS is a lower bound (kept)", file=w)
        bad = int((d["pos"] < 1 - 1e-9).sum())
        print(f"  [{tag}] POS < 1: {bad}" + ("  <-- investigate" if bad else "  (none)"), file=w)
    if "tag" in on.columns and "tag" in off.columns:
        if set(on["tag"]) != set(off["tag"]):
            print("  WARNING: on/off runs cover different instance sets!", file=w)
    print("=" * 66, file=w)

    # ---- emit -------------------------------------------------------------
    doc = "\n\n".join([
        "% Auto-generated by make_nfg_tables.py -- do not edit by hand.",
        f"% sources: {args.on.name}, {args.off.name}",
        t_summary(on),
        t_by_cap_n(on, npc),
        t_comp(on, npc),
        t_vest(on, off),
    ])
    args.out.write_text(doc + "\n")
    print(f"wrote {args.out}", file=w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
