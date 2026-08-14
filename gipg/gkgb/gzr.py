"""CEI-GZR solver for bilinear KPG with shared constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np
import time

from .instance import KPGInstance
from .model import build_master_model, KPGModel
from .best_response import solve_best_response
from .objectives import player_utility
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
    first_pne_time: Optional[float] = None
    mip_gap: Optional[float] = None
    obj_bound: Optional[float] = None


def solve_gzr(
    inst: KPGInstance,
    *,
    alpha: float = 1.0,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    br_time_limit: Optional[float] = None,
    warm_start: Optional[np.ndarray] = None,
    stop_at_first: bool = False,
    verbose: bool = False,
    tol: float = 1e-9,
) -> GZRResult:
    """Solve for an alpha-approx GNE using CEI-GZR (lazy CEIs) for bilinear KPG.

    Maximises social welfare subject to feasibility + CEIs.
    If alpha=1, seeks a welfare-optimal GNE.

    Parameters
    ----------
    warm_start : optional np.ndarray of shape (n, m)
        If provided, used as MIP start hint.
    stop_at_first : bool
        If True, terminate as soon as a verified alpha-PNE is found (FEASIBLE status).

    Returns OPTIMAL if proven, FEASIBLE if stop_at_first found one, or TIME_LIMIT.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    kpgm = build_master_model(
        inst,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        verbose=verbose,
    )
    mdl = kpgm.model
    n, m = inst.n, inst.m

    # ---- Warm-start ----
    if warm_start is not None:
        for i in range(n):
            for j in range(m):
                kpgm.x[(i, j)].Start = float(warm_start[i, j])
        # y hints
        for i in range(n):
            for k in range(n):
                if k == i:
                    continue
                for j in range(m):
                    kpgm.y[(i, k, j)].Start = float(warm_start[i, j]) * float(warm_start[k, j])
        # zfull hints
        for i in range(n):
            for j in range(m):
                opp_use = sum(int(warm_start[k, j]) for k in range(n) if k != i)
                kpgm.zfull[(i, j)].Start = 1.0 if opp_use >= int(inst.c[j]) else 0.0

    stats = {"cuts_added": 0, "br_calls": 0, "first_pne_time": None}
    start = time.time()

    def callback(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return

        # Extract incumbent profile
        X = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                X[i, j] = int(round(model.cbGetSolution(kpgm.x[(i, j)])))

        any_violation = False

        for i in range(n):
            stats["br_calls"] += 1

            # Current utility (with bilinear terms)
            pi_cur = player_utility(inst, X, i)

            # Best response
            br = solve_best_response(
                inst,
                player=i,
                x_others=X,
                time_limit=br_time_limit,
                verbose=False,
            )
            pi_hat = br.profit

            # Check alpha-approx violation: alpha * pi_cur < pi_hat
            if float(alpha) * float(pi_cur) + tol < float(pi_hat):
                any_violation = True

                # Compute big-M: upper bound on deviation profit.
                # Safe choice: use pi_hat as M (when any deviation item is full,
                # the -M*zfull term makes the constraint trivially satisfied).
                bigM = abs(pi_hat) + 1.0

                cut = CEICut(
                    player=i,
                    x_hat=br.x_hat,
                    profit_hat=float(pi_hat),
                    alpha=float(alpha),
                    bigM=bigM,
                )
                lhs, rhs = build_cei_lhs_rhs(
                    inst,
                    cut=cut,
                    x_vars=kpgm.x,
                    y_vars=kpgm.y,
                    zfull_vars=kpgm.zfull,
                )
                model.cbLazy(lhs >= rhs)
                stats["cuts_added"] += 1

        if not any_violation:
            if stats["first_pne_time"] is None:
                stats["first_pne_time"] = time.time() - start
            if stop_at_first:
                model._pne_profile = X
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
    if stop_at_first and hasattr(mdl, "_pne_profile"):
        status = "FEASIBLE"

    profile = None
    obj_val = None
    if hasattr(mdl, "_pne_profile"):
        profile = getattr(mdl, "_pne_profile")
        if mdl.SolCount > 0:
            obj_val = float(mdl.ObjVal)
    elif mdl.SolCount > 0:
        profile = kpgm.extract_profile()
        obj_val = float(mdl.ObjVal)

    mip_gap_val = float(mdl.MIPGap) if mdl.SolCount > 0 else None
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
        mip_gap=mip_gap_val,
        obj_bound=obj_bound,
    )
