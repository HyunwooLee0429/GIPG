from __future__ import annotations

from typing import Dict, Any, Optional, List
import time

from .instance import ISBPInstance
from .heuristics import run_all_heuristics, is_regular_instance
from .gzr import solve_gzr


def solve_abs(
    inst: ISBPInstance,
    *,
    eps: float = 1e-3,
    time_limit_per_call: float = 600.0,
    max_bisect_iters: int = 60,
    regular: Optional[bool] = None,
    log_to_console: int = 0,
) -> Dict[str, Any]:
    """Alpha-bisection search, heuristic-first.

    Workflow:
      0) Run heuristics first to get an upper bound alpha_init and a candidate profile.
         - If heuristic already yields alpha <= 1 + eps, treat as exact PNE and STOP (no ZR).
      1) Run ZR at alpha=1 to test exact PNE existence (lower-bound check).
      2) If not found, bisection on alpha in [1, alpha_init], using ZR calls and BR-cache reuse.

    Outputs match Experiment 2 requirements.
    """
    t0 = time.time()
    if regular is None:
        regular = is_regular_instance(inst)

    # Best responses are cached and reused across GZR calls at different alpha.
    br_cache: List[Dict[str, Any]] = []
    time_zr_total = 0.0

    # ------------------------------------------------------------
    # Step 0: heuristic upper bound (may already yield an exact PNE).
    # ------------------------------------------------------------
    heur = run_all_heuristics(inst, regular=regular)

    # Extract best alpha + best profile robustly
    best = heur.get("best", {}) if isinstance(heur, dict) else {}
    alpha_init = float(best.get("alpha", float("inf")))
    # try common keys for the profile
    best_profile = (
        best.get("x_profile")
        or best.get("profile")
        or best.get("x")
        or heur.get("x_profile") if isinstance(heur, dict) else None
    )

    trivial_from_heuristic = (alpha_init <= 1.0 + eps)

    # If heuristic already yields alpha ~ 1, we can stop (no ZR needed).
    if trivial_from_heuristic:
        return {
            "status": "FOUND",
            "alpha_init": alpha_init,
            "alpha_star": 1.0,  # treat as exact PNE (within eps)
            "alpha_ub_verified": 1.0,
            "alpha_lb_unverified": 1.0,
            "alpha_lb_verified": 1.0,
            "x_profile": best_profile,
            "regular": bool(regular),
            "heuristics": heur,
            "trivial_from_heuristic": True,
            "verified_through_binary_search": False,
            "n_bisect_iters": 0,
            "time_total": float(time.time() - t0),
            "time_zero_regret_total": 0.0,
            "br_cache": br_cache,
        }

    # If no heuristic bound is available, fall back to a GZR-only attempt.
    if not (alpha_init < float("inf")):
        zr_fallback = solve_gzr(
            inst,
            alpha=1.0,
            time_limit=time_limit_per_call,
            log_to_console=log_to_console,
            br_cache=br_cache,
            stop_at_first_pne=True,
        )
        br_cache = zr_fallback.get("br_cache", br_cache)
        time_zr_total += float(zr_fallback.get("runtime", 0.0))

        # With stop_at_first_pne=True, a found PNE yields FEASIBLE (callback
        # terminated Gurobi early) or OPTIMAL (Gurobi proved optimality).
        if zr_fallback.get("status") in ("OPTIMAL", "FEASIBLE") and zr_fallback.get("x_profile") is not None:
            return {
                "status": "FOUND",
                "alpha_init": None,
                "alpha_star": 1.0,
                "alpha_ub_verified": 1.0,
                "alpha_lb_unverified": 1.0,
                "alpha_lb_verified": 1.0,
                "x_profile": zr_fallback.get("x_profile"),
                "regular": bool(regular),
                "heuristics": heur,
                "trivial_from_heuristic": False,
                "verified_through_binary_search": False,
                "n_bisect_iters": 0,
                "time_total": float(time.time() - t0),
                "time_zero_regret_total": float(time_zr_total),
                "br_cache": br_cache,
            }

        return {
            "status": "NO_SOLUTION",
            "alpha_init": None,
            "alpha_star": float("inf"),
            "alpha_ub_verified": None,
            "alpha_lb_unverified": 1.0,
            "alpha_lb_verified": None,
            "x_profile": None,
            "regular": bool(regular),
            "heuristics": heur,
            "trivial_from_heuristic": False,
            "verified_through_binary_search": False,
            "n_bisect_iters": 0,
            "time_total": float(time.time() - t0),
            "time_zero_regret_total": float(time_zr_total),
            "br_cache": br_cache,
        }

    # ------------------------------------------------------------
    # Step 1: test for an exact PNE at alpha = 1.
    #     - If FEASIBLE/OPTIMAL with profile: PNE found, done.
    #     - If INFEASIBLE: we have a *verified* lower bound alpha >= 1.
    #     - If TIME_LIMIT: we only have an *unverified* lower bound.
    # ------------------------------------------------------------
    zr_at_1 = solve_gzr(
        inst,
        alpha=1.0,
        time_limit=time_limit_per_call,
        log_to_console=log_to_console,
        br_cache=br_cache,
        stop_at_first_pne=True,
    )
    br_cache = zr_at_1.get("br_cache", br_cache)
    time_zr_total += float(zr_at_1.get("runtime", 0.0))

    if zr_at_1.get("status") in ("OPTIMAL", "FEASIBLE") and zr_at_1.get("x_profile") is not None:
        return {
            "status": "FOUND",
            "alpha_init": alpha_init,
            "alpha_star": 1.0,
            "alpha_ub_verified": 1.0,
            "alpha_lb_unverified": 1.0,
            "alpha_lb_verified": 1.0,
            "x_profile": zr_at_1.get("x_profile"),
            "regular": bool(regular),
            "heuristics": heur,
            "trivial_from_heuristic": False,
            "verified_through_binary_search": False,
            "n_bisect_iters": 0,
            "time_total": float(time.time() - t0),
            "time_zero_regret_total": float(time_zr_total),
            "br_cache": br_cache,
        }

    # ------------------------------------------------------------
    # Step 2: bisect on alpha between the certified bounds.
    #    We always shrink the interval using the *unverified* lower bound so the
    #    process terminates under a time limit. Separately, we track a *verified*
    #    lower bound whenever Gurobi proves infeasibility.
    # ------------------------------------------------------------
    lb_verified: Optional[float] = None
    lb_unverified: float = 1.0

    # If alpha=1 was proven infeasible, this is a certified lower bound.
    if zr_at_1.get("status") == "INFEASIBLE":
        lb_verified = 1.0

    ub_verified: float = float(alpha_init)  # feasible upper bound from heuristic
    n_iters = 0

    # keep best_profile as a feasible profile at alpha_high (heuristic provides this)
    # during search, update best_profile whenever ZR finds feasibility at a smaller alpha
    while (ub_verified - lb_unverified > eps) and (n_iters < max_bisect_iters):
        n_iters += 1
        alpha_mid = 0.5 * (lb_unverified + ub_verified)

        zr_mid = solve_gzr(
            inst,
            alpha=alpha_mid,
            time_limit=time_limit_per_call,
            log_to_console=log_to_console,
            br_cache=br_cache,
            stop_at_first_pne=True,
        )
        br_cache = zr_mid.get("br_cache", br_cache)
        time_zr_total += float(zr_mid.get("runtime", 0.0))

        st = zr_mid.get("status")

        if st in ("OPTIMAL", "FEASIBLE") and zr_mid.get("x_profile") is not None:
            # PNE found at alpha_mid → shrink upper bound.
            ub_verified = alpha_mid
            best_profile = zr_mid.get("x_profile")
        else:
            # TIME_LIMIT: only an unverified lower bound.
            # INFEASIBLE: also a verified lower bound (for this alpha).
            lb_unverified = alpha_mid
            if st == "INFEASIBLE":
                lb_verified = alpha_mid if (lb_verified is None) else max(lb_verified, alpha_mid)

    # Verification requires a certified (INFEASIBLE) lower bound within eps of
    # the upper bound.  When some lower-bound iterations timed out without an
    # infeasibility certificate, only the unverified gap is small; the verified
    # gap may exceed eps.
    verified = (lb_verified is not None) and (ub_verified - lb_verified <= eps)

    return {
        "status": "FOUND",
        "alpha_init": float(alpha_init),
        "alpha_star": float(ub_verified),
        "alpha_ub_verified": float(ub_verified),
        "alpha_lb_unverified": float(lb_unverified),
        "alpha_lb_verified": (None if lb_verified is None else float(lb_verified)),
        "x_profile": best_profile,
        "regular": bool(regular),
        "heuristics": heur,
        "trivial_from_heuristic": False,
        "verified_through_binary_search": verified,
        "n_bisect_iters": int(n_iters),
        "time_total": float(time.time() - t0),
        "time_zero_regret_total": float(time_zr_total),
        "br_cache": br_cache,
    }
