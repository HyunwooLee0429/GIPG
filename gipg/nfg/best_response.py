"""Best-response oracle for a single player in a Network Formation Game.

Given opponents' edge loads, player i solves:

    min  sum_e  w_e * x_{i,e} / (opp_e + x_{i,e})
    s.t. flow conservation  A x_i = b_i
         x_{i,e} <= rem_e        (residual capacity)
         x_{i,e} in Z_>=0

We linearise by introducing binary delta[e,s] for each edge e and load level
s in {0,...,rem_e}, similarly to the ISBP/DFG best-response oracles.
The linearised cost coefficient is:
    gamma[e,s] = w_e * s / (opp_e + s)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np

from .instance import NFGInstance


@dataclass(frozen=True)
class BRResult:
    x_hat: np.ndarray       # shape (n_edges,), integer flow
    cost: Optional[float]   # BR objective value (None if infeasible)


def solve_best_response(
    inst: NFGInstance,
    *,
    player: int,
    opp_load: np.ndarray,
    time_limit: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> BRResult:
    """Solve player i's best-response IP given opponent edge loads.

    Uses a linearised formulation with binary variables delta[e,s].
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    i = int(player)
    n_edges = inst.n_edges

    # Residual capacity per edge
    rem = np.maximum(0, inst.capacities - opp_load.astype(int))
    max_s = np.minimum(rem, int(inst.demands[i]))  # upper bound per edge

    mdl = gp.Model(f"BR_p{i}")
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.Threads = 16 if threads is None else int(threads)
    if time_limit is not None:
        mdl.Params.TimeLimit = float(time_limit)

    # delta[e,s] in {0,1}: player i sends s units on edge e
    delta: Dict[Tuple[int, int], gp.Var] = {}
    for e in range(n_edges):
        for s in range(int(max_s[e]) + 1):
            delta[(e, s)] = mdl.addVar(vtype=GRB.BINARY, name=f"d[{e},{s}]")

    mdl.update()

    # Choose exactly one load level per edge
    for e in range(n_edges):
        mdl.addConstr(
            gp.quicksum(delta[(e, s)] for s in range(int(max_s[e]) + 1)) == 1,
            name=f"one[{e}]",
        )

    # Flow conservation: for each node v,
    #   sum_{e in out(v)} x_{i,e} - sum_{e in in(v)} x_{i,e} = b_i(v)
    b_i = inst.demand_vector(i)
    for v in range(inst.n_nodes):
        out_expr = gp.quicksum(
            s * delta[(e, s)]
            for e in inst.outgoing(v)
            for s in range(int(max_s[e]) + 1)
        )
        in_expr = gp.quicksum(
            s * delta[(e, s)]
            for e in inst.incoming(v)
            for s in range(int(max_s[e]) + 1)
        )
        mdl.addConstr(out_expr - in_expr == int(b_i[v]), name=f"flow[{v}]")

    # Objective: minimise Shapley cost sharing
    #   sum_e w_e * s / (opp_e + s) * delta[e,s]
    obj = gp.LinExpr()
    for e in range(n_edges):
        w_e = float(inst.edge_costs[e])
        opp_e = int(opp_load[e])
        for s in range(int(max_s[e]) + 1):
            if s == 0:
                coef = 0.0
            else:
                coef = w_e * float(s) / float(opp_e + s)
            obj.addTerms(coef, delta[(e, s)])
    mdl.setObjective(obj, GRB.MINIMIZE)

    mdl.optimize()

    x_hat = np.zeros(n_edges, dtype=int)
    if mdl.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        for e in range(n_edges):
            for s in range(int(max_s[e]) + 1):
                if delta[(e, s)].X > 0.5:
                    x_hat[e] = s
                    break
        return BRResult(x_hat=x_hat, cost=float(mdl.ObjVal))

    # Infeasible (e.g. no feasible flow with residual capacities)
    return BRResult(x_hat=x_hat, cost=None)
