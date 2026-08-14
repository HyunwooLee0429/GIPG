"""Tightest Alpha Search (ABS) for Network Formation Games.

Binary-searches over alpha using GZR as a feasibility oracle.
Monotonicity: if an alpha-GPNE exists, then a beta-GPNE exists for beta >= alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import time

import numpy as np

from .instance import NFGInstance
from .gzr import solve_gzr, GZRResult
from .heuristics import alpha_of_profile, brd_random_restart


@dataclass
class ABSResult:
    status: str
    runtime: float
    alpha_star: Optional[float]
    alpha_ub_verified: Optional[float]
    alpha_lb_verified: Optional[float]
    alpha_lb_unverified: Optional[float]
    verified_through_binary_search: bool
    n_bisect_iters: int
    history: List[Dict[str, Any]]
    last_gzr: Optional[GZRResult]


def solve_abs(
    inst: NFGInstance,
    *,
    eps: float = 1e-2,
    alpha_low: float = 1.0,
    alpha_high: Optional[float] = None,
    time_limit_total: Optional[float] = None,
    gzr_time_limit: Optional[float] = None,
    gzr_threads: Optional[int] = None,
    br_time_limit: Optional[float] = None,
    verbose: bool = False,
    max_iter: int = 40,
    use_brd_init: bool = True,
) -> ABSResult:
    """Tightest-Alpha Search for NFG.

    Steps:
      1. (Optional) Run RRR-BRD to get an initial upper bound on alpha.
      2. Check alpha=1 (exact GPNE).
      3. Binary search on [1, alpha_high].
    """
    # Statuses indicating GZR found a feasible alpha-PNE.
    # OPTIMAL = Gurobi proved optimality; FEASIBLE = stop_at_first callback
    # found a verified PNE and terminated early.
    _PNE_FOUND = {"OPTIMAL", "FEASIBLE"}

    t0 = time.time()
    history: List[Dict[str, Any]] = []

    # --- Step 0: upper bound from BRD ---
    if alpha_high is None and use_brd_init:
        brd_profile, brd_pne, brd_time = brd_random_restart(
            inst, max_init=3, max_round=15, seed=0,
            br_time_limit=br_time_limit,
        )
        if brd_pne:
            # Exact GPNE found by BRD
            return ABSResult(
                status="OPTIMAL",
                runtime=time.time() - t0,
                alpha_star=1.0,
                alpha_ub_verified=1.0,
                alpha_lb_verified=1.0,
                alpha_lb_unverified=1.0,
                verified_through_binary_search=False,
                n_bisect_iters=0,
                history=[{"phase": "brd_pne", "runtime": brd_time}],
                last_gzr=None,
            )
        a_brd = alpha_of_profile(inst, brd_profile, br_time_limit=br_time_limit)
        alpha_high = max(alpha_low, a_brd)
        history.append({
            "phase": "brd_init",
            "alpha_from_brd": a_brd,
            "runtime": brd_time,
        })

    if alpha_high is None:
        # Fallback: use GZR with very large alpha to find any feasible profile
        res0 = solve_gzr(
            inst, alpha=1e9,
            time_limit=gzr_time_limit, threads=gzr_threads,
            br_time_limit=br_time_limit, verbose=verbose,
        )
        if res0.profile is None:
            return ABSResult(
                status="FAILED_NO_SOLUTION",
                runtime=time.time() - t0,
                alpha_star=None,
                alpha_ub_verified=None,
                alpha_lb_verified=None,
                alpha_lb_unverified=None,
                verified_through_binary_search=False,
                n_bisect_iters=0,
                history=history,
                last_gzr=res0,
            )
        a0 = alpha_of_profile(inst, res0.profile, br_time_limit=br_time_limit)
        alpha_high = max(alpha_low, a0)
        history.append({"phase": "init_upper", "alpha_high": alpha_high, "gzr_status": res0.status})

    # ------------------------------------------------------------------
    # Verified / unverified bound tracking (same logic as ISBP alpha_search)
    # ------------------------------------------------------------------
    lb_verified: Optional[float] = None
    lb_unverified: float = float(alpha_low)
    ub_verified: float = float(alpha_high)

    lo = float(alpha_low)
    hi = float(alpha_high)
    last_gzr: Optional[GZRResult] = None

    # --- Step 1: check alpha=1 ---
    if lo == 1.0:
        res1 = solve_gzr(
            inst, alpha=1.0,
            time_limit=gzr_time_limit, threads=gzr_threads,
            br_time_limit=br_time_limit, verbose=verbose,
        )
        history.append({
            "phase": "check_alpha1", "alpha": 1.0,
            "status": res1.status, "cuts": res1.cuts_added,
        })
        last_gzr = res1

        # Track verified/unverified bounds from alpha=1 check
        if res1.status == "INFEASIBLE":
            lb_verified = 1.0
        # TIME_LIMIT at alpha=1: only unverified lower bound (lb_unverified already = 1.0)

        if res1.status in _PNE_FOUND:
            return ABSResult(
                status="OPTIMAL",
                runtime=time.time() - t0,
                alpha_star=1.0,
                alpha_ub_verified=1.0,
                alpha_lb_verified=1.0,
                alpha_lb_unverified=1.0,
                verified_through_binary_search=False,
                n_bisect_iters=0,
                history=history,
                last_gzr=res1,
            )

    # --- Step 2: binary search ---
    n_iters = 0
    for it in range(max_iter):
        n_iters = it
        if time_limit_total is not None and (time.time() - t0) >= float(time_limit_total):
            verified = (lb_verified is not None) and (ub_verified - lb_verified <= eps)
            return ABSResult(
                status="TIME_LIMIT_TOTAL",
                runtime=time.time() - t0,
                alpha_star=float(ub_verified) if ub_verified is not None else None,
                alpha_ub_verified=ub_verified,
                alpha_lb_verified=lb_verified,
                alpha_lb_unverified=lb_unverified,
                verified_through_binary_search=verified,
                n_bisect_iters=n_iters,
                history=history,
                last_gzr=last_gzr,
            )

        if hi - lo <= eps * max(1.0, lo):
            break

        mid = 0.5 * (lo + hi)
        res = solve_gzr(
            inst, alpha=mid,
            time_limit=gzr_time_limit, threads=gzr_threads,
            br_time_limit=br_time_limit, verbose=verbose,
        )
        last_gzr = res
        history.append({
            "iter": it, "alpha": mid,
            "status": res.status, "cuts": res.cuts_added,
        })

        if res.status in _PNE_FOUND:
            hi = mid
            ub_verified = mid
        elif res.status == "INFEASIBLE":
            lo = mid
            lb_unverified = mid
            lb_verified = mid if (lb_verified is None) else max(lb_verified, mid)
        else:
            # TIME_LIMIT: only an unverified lower bound; continue bisection.
            lo = mid
            lb_unverified = mid

    # --- Final: confirm hi ---
    res_final = solve_gzr(
        inst, alpha=hi,
        time_limit=gzr_time_limit, threads=gzr_threads,
        br_time_limit=br_time_limit, verbose=verbose,
    )
    history.append({"phase": "final", "alpha": hi, "status": res_final.status})
    last_gzr = res_final

    # Verification requires a certified (INFEASIBLE) lower bound within eps of
    # the upper bound.  When some lower-bound iterations timed out without an
    # infeasibility certificate, only the unverified gap is small; the verified
    # gap may exceed eps.
    verified = (lb_verified is not None) and (ub_verified - lb_verified <= eps)

    if res_final.status in _PNE_FOUND:
        return ABSResult(
            status="OPTIMAL",
            runtime=time.time() - t0,
            alpha_star=float(hi),
            alpha_ub_verified=float(ub_verified),
            alpha_lb_verified=(None if lb_verified is None else float(lb_verified)),
            alpha_lb_unverified=float(lb_unverified),
            verified_through_binary_search=verified,
            n_bisect_iters=n_iters,
            history=history,
            last_gzr=res_final,
        )

    return ABSResult(
        status="FAILED",
        runtime=time.time() - t0,
        alpha_star=None,
        alpha_ub_verified=float(ub_verified),
        alpha_lb_verified=(None if lb_verified is None else float(lb_verified)),
        alpha_lb_unverified=float(lb_unverified),
        verified_through_binary_search=False,
        n_bisect_iters=n_iters,
        history=history,
        last_gzr=res_final,
    )
