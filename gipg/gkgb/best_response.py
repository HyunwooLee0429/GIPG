"""Best-response oracle for bilinear KPG with shared constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np

from .instance import KPGInstance


@dataclass(frozen=True)
class BestResponse:
    x_hat: np.ndarray   # shape (m,), binary
    profit: float        # BR profit (includes interaction terms)


def solve_best_response(
    inst: KPGInstance,
    *,
    player: int,
    x_others: np.ndarray,   # shape (n, m) full profile; will ignore row `player`
    time_limit: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> BestResponse:
    """Solve player i's best response given opponents' decisions.

    Player i maximises:
        sum_j [p[i,j] + sum_{k!=i} C[i,k,j] * x_others[k,j]] * x[i,j]
    subject to:
        knapsack:     sum_j w[i,j]*x[i,j] <= b[i]
        availability: x[i,j] = 0  if  sum_{k!=i} x_others[k,j] >= c[j]

    Given fixed x_others, the bilinear terms reduce to a standard linear
    knapsack with modified profits.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    n, m = inst.n, inst.m
    i = int(player)

    # Compute opponent usage and availability
    opp_use = x_others.sum(axis=0) - x_others[i, :]          # shape (m,)
    avail = (opp_use <= (inst.c - 1)).astype(int)              # shape (m,)

    # Modified profits: p[i,j] + sum_{k!=i} C[i,k,j] * x_others[k,j]
    mod_profit = inst.p[i].astype(float).copy()
    for k in range(n):
        if k == i:
            continue
        mod_profit += inst.C[i, k].astype(float) * x_others[k].astype(float)

    mdl = gp.Model(f"BR_{i}")
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.Threads = 16 if threads is None else int(threads)
    if time_limit is not None:
        mdl.Params.TimeLimit = float(time_limit)

    x = {j: mdl.addVar(vtype=GRB.BINARY, name=f"x[{j}]") for j in range(m)}
    mdl.update()

    # Knapsack constraint
    mdl.addConstr(
        gp.quicksum(int(inst.w[i, j]) * x[j] for j in range(m)) <= int(inst.b[i]),
        name="knap",
    )

    # Availability: fix unavailable items to 0
    for j in range(m):
        if avail[j] == 0:
            mdl.addConstr(x[j] == 0, name=f"avail[{j}]")

    # Objective: maximise modified profit
    obj = gp.quicksum(float(mod_profit[j]) * x[j] for j in range(m))
    mdl.ModelSense = GRB.MAXIMIZE
    mdl.setObjective(obj)

    mdl.optimize()

    if mdl.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return BestResponse(x_hat=np.zeros(m, dtype=int), profit=0.0)

    x_hat = np.array([int(round(x[j].X)) for j in range(m)], dtype=int)
    profit = float(sum(float(mod_profit[j]) * x_hat[j] for j in range(m)))
    return BestResponse(x_hat=x_hat, profit=profit)
