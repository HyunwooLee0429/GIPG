from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np

from .instance import GKGInstance


@dataclass(frozen=True)
class BestResponse:
    x_hat: np.ndarray  # shape (m,), binary
    profit: int


def solve_best_response(
    inst: GKGInstance,
    *,
    player: int,
    x_others: np.ndarray,  # shape (n,m) full profile, will ignore row player
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> BestResponse:
    """Solve player's best response ILP given opponents decisions x_others.

    Availability: item j is available for i if sum_{k!=i} x_{k,j} <= c_j - 1
    (since i would add at most 1 unit).
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    n, m = inst.n, inst.m
    i = int(player)
    opp_use = x_others.sum(axis=0) - x_others[i, :]
    avail = (opp_use <= (inst.c - 1)).astype(int)

    mdl = gp.Model(f"BR_{i}")
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.Threads = 16 if threads is None else int(threads)
    if time_limit is not None:
        mdl.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        mdl.Params.MIPGap = float(mip_gap)

    x = {j: mdl.addVar(vtype=GRB.BINARY, name=f"x[{j}]") for j in range(m)}
    mdl.update()

    # knapsack
    mdl.addConstr(gp.quicksum(int(inst.w[i, j]) * x[j] for j in range(m)) <= int(inst.b[i]))

    # availability
    for j in range(m):
        if avail[j] == 0:
            mdl.addConstr(x[j] == 0)

    obj = gp.quicksum(int(inst.p[i, j]) * x[j] for j in range(m))
    mdl.ModelSense = GRB.MAXIMIZE
    mdl.setObjective(obj)

    mdl.optimize()

    if mdl.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        # If infeasible (shouldn't happen), return zero
        return BestResponse(x_hat=np.zeros(m, dtype=int), profit=0)

    x_hat = np.array([int(round(x[j].X)) for j in range(m)], dtype=int)
    profit = int(round(sum(int(inst.p[i, j]) * x_hat[j] for j in range(m))))
    return BestResponse(x_hat=x_hat, profit=profit)
