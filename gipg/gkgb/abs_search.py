"""Alpha-bisection search (ABS) for bilinear KPG using GZR as feasibility oracle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import time

from .instance import KPGInstance
from .gzr import solve_gzr, GZRResult
from .best_response import solve_best_response
from .objectives import player_utility


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


def alpha_needed_for_profile(inst: KPGInstance, X: np.ndarray) -> float:
    """Compute alpha required for X to be an alpha-approx GNE (maximisation):
        alpha * u_i(X) >= u_i(BR_i, X_{-i}) for all i.

    Returns max_i (pi_hat / max(eps, pi_cur)).
    """
    n, m = inst.n, inst.m
    ratios = []
    for i in range(n):
        br = solve_best_response(inst, player=i, x_others=X, verbose=False)
        pi_hat = float(br.profit)
        pi_cur = player_utility(inst, X, i)
        if pi_cur <= 0.0:
            ratios.append(float("inf") if pi_hat > 0 else 1.0)
        else:
            ratios.append(pi_hat / pi_cur)
    return float(max(ratios)) if ratios else 1.0


def solve_abs(
    inst: KPGInstance,
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
) -> ABSResult:
    """Tightest-Alpha Search for bilinear KPG.

    Uses GZR as a feasibility oracle with binary bisection on alpha.
    Tracks verified (INFEASIBLE) and unverified (TIME_LIMIT) lower bounds.
    """
    _PNE_FOUND = {"OPTIMAL", "FEASIBLE"}

    t0 = time.time()
    history: List[Dict[str, Any]] = []

    # Step 0: Initialise upper bound
    if alpha_high is None:
        big_alpha = 1e9
        res0 = solve_gzr(
            inst,
            alpha=big_alpha,
            time_limit=gzr_time_limit,
            threads=gzr_threads,
            br_time_limit=br_time_limit,
            stop_at_first=True,
            verbose=verbose,
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
        a0 = alpha_needed_for_profile(inst, res0.profile)
        alpha_high = max(alpha_low, a0)
        history.append({"phase": "init_upper", "alpha_high": alpha_high, "alpha_from_profile": a0, "gzr_status": res0.status})

    lb_verified: Optional[float] = None
    lb_unverified: float = float(alpha_low)
    ub_verified: float = float(alpha_high)

    lo = float(alpha_low)
    hi = float(alpha_high)
    last_gzr: Optional[GZRResult] = None

    # Step 1: Check alpha=1
    if lo == 1.0:
        res1 = solve_gzr(
            inst,
            alpha=1.0,
            time_limit=gzr_time_limit,
            threads=gzr_threads,
            br_time_limit=br_time_limit,
            stop_at_first=True,
            verbose=verbose,
        )
        history.append({"phase": "check_alpha1", "alpha": 1.0, "status": res1.status, "obj": res1.obj_val, "cuts": res1.cuts_added})
        last_gzr = res1

        if res1.status == "INFEASIBLE":
            lb_verified = 1.0

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

    # Step 2: Binary search
    n_iters = 0
    for it in range(max_iter):
        n_iters = it
        if time_limit_total is not None and (time.time() - t0) >= float(time_limit_total):
            verified = (lb_verified is not None) and (ub_verified - lb_verified <= eps)
            return ABSResult(
                status="TIME_LIMIT_TOTAL",
                runtime=time.time() - t0,
                alpha_star=float(ub_verified),
                alpha_ub_verified=ub_verified,
                alpha_lb_verified=lb_verified,
                alpha_lb_unverified=lb_unverified,
                verified_through_binary_search=verified,
                n_bisect_iters=it,
                history=history,
                last_gzr=last_gzr,
            )

        if hi - lo <= eps * max(1.0, lo):
            break

        mid = 0.5 * (lo + hi)

        res = solve_gzr(
            inst,
            alpha=mid,
            time_limit=gzr_time_limit,
            threads=gzr_threads,
            br_time_limit=br_time_limit,
            stop_at_first=True,
            verbose=verbose,
        )
        last_gzr = res
        history.append({"iter": it, "alpha": mid, "status": res.status, "obj": res.obj_val, "cuts": res.cuts_added})

        if res.status in _PNE_FOUND:
            hi = mid
            ub_verified = mid
        elif res.status == "INFEASIBLE":
            lo = mid
            lb_unverified = mid
            lb_verified = mid if (lb_verified is None) else max(lb_verified, mid)
        else:
            # TIME_LIMIT: unverified lower bound
            lo = mid
            lb_unverified = mid

    # Step 3: Final refine
    res_final = solve_gzr(
        inst,
        alpha=hi,
        time_limit=gzr_time_limit,
        threads=gzr_threads,
        br_time_limit=br_time_limit,
        stop_at_first=True,
        verbose=verbose,
    )
    history.append({"phase": "final", "alpha": hi, "status": res_final.status, "obj": res_final.obj_val, "cuts": res_final.cuts_added})
    last_gzr = res_final

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
