from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np
import time

from .instance import GKGInstance
from .model import build_master_model, GKGModel
from .best_response import solve_best_response
from .cuts import CEICut, build_cei_lhs_rhs


@dataclass
class GZRResult:
    status: str
    runtime: float
    obj_val: Optional[float]
    profile: Optional[np.ndarray]
    cuts_added: int
    br_calls: int
    alpha: float
    first_pne_time: Optional[float] = None  # wall-clock seconds to first verified PNE
    mip_gap: Optional[float] = None         # Gurobi MIPGap (None if unavailable)
    obj_bound: Optional[float] = None       # Gurobi ObjBound (None if unavailable)


def solve_gzr(
    inst: GKGInstance,
    *,
    alpha: float = 1.0,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    br_time_limit: Optional[float] = None,
    br_threads: Optional[int] = None,
    warm_start: Optional[np.ndarray] = None,
    stop_at_first: bool = False,
    verbose: bool = False,
    tol: float = 1e-9,
) -> GZRResult:
    """Solve for an alpha-approx GNE using CEI-GZR (lazy CEIs).

    We maximize social welfare subject to feasibility + CEIs. If alpha=1, this seeks a welfare-optimal GNE.

    Parameters
    ----------
    warm_start : optional np.ndarray of shape (n, m)
        If provided, used as MIP start hint.
    stop_at_first : bool
        If True, terminate as soon as a verified PNE is found (FEASIBLE status).
        If False (default), optimise to MIP gap=0 to find the best PNE.

    Returns OPTIMAL if proven, FEASIBLE if stop_at_first found one, or TIME_LIMIT if timed out.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    gkgm = build_master_model(
        inst,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        verbose=verbose,
    )
    mdl = gkgm.model
    n, m = inst.n, inst.m

    # --- Warm-start ---
    if warm_start is not None:
        for i in range(n):
            for j in range(m):
                gkgm.x[(i, j)].Start = float(warm_start[i, j])
        # Also set zfull hints from the warm-start profile
        for i in range(n):
            for j in range(m):
                opp_use = sum(int(warm_start[k, j]) for k in range(n) if k != i)
                gkgm.zfull[(i, j)].Start = 1.0 if opp_use >= int(inst.c[j]) else 0.0

    stats = {"cuts_added": 0, "br_calls": 0, "first_pne_time": None}
    start = time.time()

    def callback(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return

        # incumbent profile
        X = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                X[i, j] = int(round(model.cbGetSolution(gkgm.x[(i, j)])))

        any_violation = False

        # For each player: best response & add violated CEI
        for i in range(n):
            stats["br_calls"] += 1
            br = solve_best_response(
                inst,
                player=i,
                x_others=X,
                time_limit=br_time_limit,
                threads=br_threads,
                verbose=False,
            )
            pi_cur = int(sum(int(inst.p[i, j]) * int(X[i, j]) for j in range(m)))
            pi_hat = int(br.profit)

            # Check alpha-approx violation: alpha*pi_cur < pi_hat
            if float(alpha) * float(pi_cur) + tol < float(pi_hat):
                any_violation = True
                # Build and add lazy CEI cut
                bigM = float(pi_hat)  # safe
                cut = CEICut(player=i, x_hat=br.x_hat, profit_hat=pi_hat, alpha=float(alpha), bigM=bigM)
                lhs, rhs = build_cei_lhs_rhs(inst, cut=cut, x_vars=gkgm.x, zfull_vars=gkgm.zfull)
                model.cbLazy(lhs >= rhs)
                stats["cuts_added"] += 1

        if not any_violation:
            # Record time of first verified PNE (only once)
            if stats["first_pne_time"] is None:
                stats["first_pne_time"] = time.time() - start
            if stop_at_first:
                model._gpne_profile = X
                model.terminate()

    mdl.optimize(callback)

    runtime = time.time() - start

    if mdl.Status == GRB.OPTIMAL:
        status = "OPTIMAL"
    elif mdl.Status == GRB.TIME_LIMIT:
        status = "TIME_LIMIT"
    elif mdl.Status == GRB.INTERRUPTED:
        status = "INTERRUPTED" if not stop_at_first else "FEASIBLE"
    elif mdl.Status == GRB.INFEASIBLE:
        status = "INFEASIBLE"
    else:
        status = f"STATUS_{mdl.Status}"

    # If stop_at_first and we stored a verified profile, use FEASIBLE status
    if stop_at_first and hasattr(mdl, "_gpne_profile"):
        status = "FEASIBLE"

    profile = None
    obj_val = None
    if hasattr(mdl, "_gpne_profile"):
        profile = getattr(mdl, "_gpne_profile")
        if mdl.SolCount > 0:
            obj_val = float(mdl.ObjVal)
    elif mdl.SolCount > 0:
        profile = gkgm.extract_profile()
        obj_val = float(mdl.ObjVal)

    # MIPGap (requires at least one feasible solution).
    mip_gap = float(mdl.MIPGap) if mdl.SolCount > 0 else None
    # ObjBound — always attempt (useful even when SolCount == 0).
    try:
        obj_bound = float(mdl.ObjBound)
    except Exception:
        obj_bound = None

    return GZRResult(
        status=status,
        runtime=float(runtime),
        obj_val=obj_val,
        profile=profile,
        cuts_added=int(stats["cuts_added"]),
        br_calls=int(stats["br_calls"]),
        alpha=float(alpha),
        first_pne_time=stats["first_pne_time"],
        mip_gap=mip_gap,
        obj_bound=obj_bound,
    )
