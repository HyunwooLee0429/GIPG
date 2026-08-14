"""Social optimum solver for bilinear KPG (maximise welfare without equilibrium constraints)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import time

from .instance import KPGInstance
from .model import build_master_model


@dataclass
class SOResult:
    status: str
    opt_cost: Optional[float]     # social welfare at optimum (higher = better)
    runtime: float
    profile: Optional[np.ndarray]


def solve_social_optimum(
    inst: KPGInstance,
    *,
    time_limit: float = 600.0,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> SOResult:
    """Solve the social optimum: max total welfare WITHOUT CEI constraints.

    The cooperative benchmark — all players coordinated to maximise
    total profit subject to knapsack and shared-capacity constraints only.
    """
    kpgm = build_master_model(
        inst,
        time_limit=time_limit,
        threads=threads,
        verbose=verbose,
    )
    mdl = kpgm.model
    mdl.Params.LazyConstraints = 0

    t0 = time.time()
    mdl.optimize()
    runtime = time.time() - t0

    from gurobipy import GRB

    if mdl.Status == GRB.OPTIMAL:
        status = "OPTIMAL"
    elif mdl.Status == GRB.TIME_LIMIT:
        status = "TIME_LIMIT"
    elif mdl.Status == GRB.INFEASIBLE:
        status = "INFEASIBLE"
    else:
        status = f"STATUS_{mdl.Status}"

    profile = None
    opt_cost = None
    if mdl.SolCount > 0:
        profile = kpgm.extract_profile()
        opt_cost = float(mdl.ObjVal)

    return SOResult(
        status=status,
        opt_cost=opt_cost,
        runtime=float(runtime),
        profile=profile,
    )


def compute_pos(so_welfare: float, best_pne_welfare: float) -> float:
    """Price of Stability for maximisation games.

    POS = SO_welfare / best_PNE_welfare >= 1.
    Closer to 1 = less inefficiency.
    """
    if best_pne_welfare <= 0:
        return float("inf")
    return so_welfare / best_pne_welfare
