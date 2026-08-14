"""Master MILP for bilinear KPG with linearised interaction terms and shared constraints."""
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

from .instance import KPGInstance


@dataclass
class KPGModel:
    inst: KPGInstance
    model: "gp.Model"
    x: Dict[Tuple[int, int], "gp.Var"]          # (i,j) -> binary
    y: Dict[Tuple[int, int, int], "gp.Var"]      # (i,k,j) -> binary, linearisation of x[i,j]*x[k,j]
    zfull: Dict[Tuple[int, int], "gp.Var"]       # (i,j) -> binary, item j full w.r.t. player i

    def extract_profile(self) -> np.ndarray:
        n, m = self.inst.n, self.inst.m
        X = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                X[i, j] = int(round(self.x[(i, j)].X))
        return X


def build_master_model(
    inst: KPGInstance,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> KPGModel:
    """Build the master MILP for bilinear KPG.

    Variables:
      x[i,j]      binary — player i selects item j
      y[i,k,j]    binary — linearisation of x[i,j] * x[k,j] for ordered (i,k), i!=k
      zfull[i,j]  binary — item j is full from player i's perspective

    Constraints:
      - Knapsack:      sum_j w[i,j]*x[i,j] <= b[i]
      - Shared cap:    sum_i x[i,j] <= c[j]
      - McCormick:     y[i,k,j] <= x[i,j];  y[i,k,j] <= x[k,j];
                       y[i,k,j] >= x[i,j] + x[k,j] - 1
      - zfull:         zfull[i,j] = 1  <=>  sum_{k!=i} x[k,j] >= c[j]

    Objective:
      max sum_i [sum_j p[i,j]*x[i,j] + sum_{k!=i} sum_j C[i,k,j]*y[i,k,j]]
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    n, m = inst.n, inst.m
    mdl = gp.Model("KPG_bilinear_master")
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.LazyConstraints = 1
    mdl.Params.Threads = 16 if threads is None else int(threads)
    if time_limit is not None:
        mdl.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        mdl.Params.MIPGap = float(mip_gap)

    # ---- Decision variables ----
    x: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(n):
        for j in range(m):
            x[(i, j)] = mdl.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")

    # Linearisation variables y[i,k,j] for ordered pairs (i,k), i!=k
    y: Dict[Tuple[int, int, int], gp.Var] = {}
    for i in range(n):
        for k in range(n):
            if k == i:
                continue
            for j in range(m):
                y[(i, k, j)] = mdl.addVar(vtype=GRB.BINARY, name=f"y[{i},{k},{j}]")

    # zfull indicators
    zfull: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(n):
        for j in range(m):
            zfull[(i, j)] = mdl.addVar(vtype=GRB.BINARY, name=f"zfull[{i},{j}]")

    mdl.update()

    # ---- Constraints ----

    # Private knapsack constraints
    for i in range(n):
        mdl.addConstr(
            gp.quicksum(int(inst.w[i, j]) * x[(i, j)] for j in range(m)) <= int(inst.b[i]),
            name=f"knap[{i}]",
        )

    # Shared item capacities
    for j in range(m):
        mdl.addConstr(
            gp.quicksum(x[(i, j)] for i in range(n)) <= int(inst.c[j]),
            name=f"cap[{j}]",
        )

    # McCormick linearisation: y[i,k,j] = x[i,j] * x[k,j]
    for i in range(n):
        for k in range(n):
            if k == i:
                continue
            for j in range(m):
                mdl.addConstr(y[(i, k, j)] <= x[(i, j)], name=f"mc_a[{i},{k},{j}]")
                mdl.addConstr(y[(i, k, j)] <= x[(k, j)], name=f"mc_b[{i},{k},{j}]")
                mdl.addConstr(y[(i, k, j)] >= x[(i, j)] + x[(k, j)] - 1, name=f"mc_c[{i},{k},{j}]")

    # zfull indicator constraints
    for i in range(n):
        for j in range(m):
            opp_use = gp.quicksum(x[(k, j)] for k in range(n) if k != i)
            cj = int(inst.c[j])

            # (A) full => zfull=1
            mdl.addConstr(
                opp_use <= (cj - 1) + (n - 1) * zfull[(i, j)],
                name=f"zfull_A[{i},{j}]",
            )

            # (B) zfull=1 => full
            mdl.addConstr(
                opp_use >= cj * zfull[(i, j)],
                name=f"zfull_B[{i},{j}]",
            )

    # ---- Objective: maximise social welfare ----
    obj = gp.LinExpr()
    # Direct profits
    for i in range(n):
        for j in range(m):
            obj.addTerms(float(inst.p[i, j]), x[(i, j)])
    # Interaction terms (bilinear, linearised)
    for i in range(n):
        for k in range(n):
            if k == i:
                continue
            for j in range(m):
                coeff = float(inst.C[i, k, j])
                if coeff != 0.0:
                    obj.addTerms(coeff, y[(i, k, j)])

    mdl.ModelSense = GRB.MAXIMIZE
    mdl.setObjective(obj)

    return KPGModel(inst=inst, model=mdl, x=x, y=y, zfull=zfull)
