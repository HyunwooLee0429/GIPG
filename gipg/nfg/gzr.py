"""Generalized Zero-Regret (GZR) algorithm for Network Formation Games.

Uses the discretised master model with lazy CEI constraints.
At each MIPSOL callback:
  1. Extract incumbent integer profile.
  2. For each player, solve best-response IP.
  3. If alpha-violation found, add CEI lazy cut.
  4. If no violation for any player, terminate (alpha-GPNE found).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple, Set
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np

from .instance import NFGInstance
from .model import build_master_model, NFGModel
from .best_response import solve_best_response
from .objectives import player_cost, edge_loads, player_cost_against_opponents
from .cuts import CEICut, build_cei_lhs_rhs, build_cei_with_opponent_load


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
    inst: NFGInstance,
    *,
    alpha: float = 1.0,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    br_time_limit: Optional[float] = None,
    verbose: bool = False,
    tol: float = 1e-9,
    use_opponent_load_cei: bool = True,
    stop_at_first: bool = True,
    warm_start: Optional[np.ndarray] = None,
    add_vest_cut: bool = True,
) -> GZRResult:
    """Solve for an alpha-approx GPNE using CEI-GZR (lazy CEIs).

    Minimises social cost subject to feasibility + lazy CEIs.

    Parameters
    ----------
    warm_start : optional (n_players, n_edges) int array
        If provided, seeds Gurobi's MIP Start with this profile.
    add_vest_cut : bool
        If True (default), add weaker VEST cuts based on fundamental
        cycle basis to the master model.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    nfgm = build_master_model(
        inst,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        verbose=verbose,
        add_vest_cut=add_vest_cut,
    )
    mdl = nfgm.model
    n = inst.n_players
    E = inst.n_edges

    stats = {"cuts_added": 0, "br_calls": 0, "first_pne_time": None}
    added_sigs: Set[Tuple[int, ...]] = set()
    start = time.time()

    def _br_sig(player: int, x_hat: np.ndarray) -> Tuple[int, ...]:
        return (int(player),) + tuple(int(v) for v in x_hat)

    def callback(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return

        # Extract incumbent
        X = np.zeros((n, E), dtype=int)
        for i in range(n):
            for e in range(E):
                X[i, e] = int(round(model.cbGetSolution(nfgm.x[(i, e)])))

        loads = edge_loads(inst, X)
        any_violation = False

        for i in range(n):
            stats["br_calls"] += 1
            opp = loads - X[i]
            br = solve_best_response(
                inst, player=i, opp_load=opp,
                time_limit=br_time_limit, verbose=False,
            )
            if br.cost is None:
                continue

            curr = player_cost(inst, X, i)

            # Check alpha-approx violation: curr > alpha * br_cost + tol
            if curr <= alpha * br.cost + tol:
                continue

            any_violation = True
            sig = _br_sig(i, br.x_hat)
            if sig in added_sigs:
                continue

            # Compute bigM: safe upper bound on cost difference
            bigM = float(curr) + 1.0

            cut = CEICut(
                player=i,
                x_hat=br.x_hat,
                cost_hat=float(br.cost),
                alpha=float(alpha),
                bigM=bigM,
            )

            if use_opponent_load_cei:
                lhs, rhs = build_cei_with_opponent_load(
                    inst, cut=cut, y_vars=nfgm.y,
                )
            else:
                lhs, rhs = build_cei_lhs_rhs(
                    inst, cut=cut, y_vars=nfgm.y,
                )

            model.cbLazy(lhs <= rhs)
            added_sigs.add(sig)
            stats["cuts_added"] += 1

        if not any_violation:
            # Record time of first verified PNE (only once)
            if stats["first_pne_time"] is None:
                stats["first_pne_time"] = time.time() - start
            if stop_at_first:
                model._gpne_profile = X
                model.terminate()

    # Seed Gurobi MIP Start with warm_start profile if provided
    if warm_start is not None:
        for i in range(n):
            for e in range(E):
                nfgm.x[(i, e)].Start = int(warm_start[i, e])

    mdl.optimize(callback)

    runtime = time.time() - start

    # Determine status
    profile = None
    obj_val = None

    if hasattr(mdl, "_gpne_profile"):
        profile = getattr(mdl, "_gpne_profile")
    elif mdl.SolCount > 0:
        profile = nfgm.extract_profile()

    if profile is not None:
        obj_val = float(mdl.ObjVal) if mdl.SolCount > 0 else None

    # Map Gurobi status faithfully.
    # When stop_at_first=True the callback calls model.terminate() after
    # finding a verified PNE, so Gurobi reports USER_OBJ_LIMIT or
    # INTERRUPTED — treat that as OPTIMAL (we got what we asked for).
    # When stop_at_first=False we need Gurobi to prove global optimality
    # (MIP gap = 0), so a feasible PNE under TIME_LIMIT is NOT optimal.
    grb_status = mdl.Status

    if hasattr(mdl, "_gpne_profile"):
        # stop_at_first path: callback terminated after finding a verified PNE,
        # but it is not necessarily the best (minimum social cost) PNE.
        status = "FEASIBLE"
    elif grb_status == GRB.OPTIMAL:
        # Gurobi proved optimality (MIP gap = 0)
        status = "OPTIMAL"
    elif grb_status == GRB.INFEASIBLE:
        status = "INFEASIBLE"
    elif grb_status == GRB.TIME_LIMIT:
        # May or may not have a feasible PNE — profile is still returned
        status = "TIME_LIMIT"
    else:
        status = f"STATUS_{grb_status}"

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
