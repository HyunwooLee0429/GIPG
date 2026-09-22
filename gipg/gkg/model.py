from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np

from .instance import GKGInstance


@dataclass
class GKGModel:
    inst: GKGInstance
    model: "gp.Model"
    x: Dict[Tuple[int, int], "gp.Var"]     # (i,j) -> var
    zfull: Dict[Tuple[int, int], "gp.Var"] # (i,j) -> var indicating item j full w.r.t. i

    def extract_profile(self) -> np.ndarray:
        n, m = self.inst.n, self.inst.m
        X = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                v = self.x[(i, j)].X
                X[i, j] = int(round(v))
        return X


def build_master_model(
    inst: GKGInstance,
    *,
    sense: str = "max",
    objective: str = "social_welfare",
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> GKGModel:
    """Build the master MILP for GKG.

    We include:
      - decision vars x_{ij} in {0,1}
      - helper vars zfull_{ij} in {0,1} indicating whether item j is already full
        (from the perspective of player i), i.e.
            zfull_{i,j} = 1  <=>  sum_{k != i} x_{k,j} >= c_j

    The helper vars allow CEI activation without adding new variables in callbacks.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    n, m = inst.n, inst.m
    mdl = gp.Model("GKG_master")
    mdl.Params.OutputFlag = 1 if verbose else 0
    if log_file is not None:
        # Full Gurobi log to file (node log, root bound, callback timing) without console output.
        mdl.Params.OutputFlag = 1
        mdl.Params.LogToConsole = 1 if verbose else 0
        mdl.Params.LogFile = str(log_file)
    mdl.Params.LazyConstraints = 1
    mdl.Params.Threads = 16 if threads is None else int(threads)
    if time_limit is not None:
        mdl.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        mdl.Params.MIPGap = float(mip_gap)
    if seed is not None:
        mdl.Params.Seed = int(seed)

    x: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(n):
        for j in range(m):
            x[(i, j)] = mdl.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")

    # zfull_{i,j}: =1 iff opponents already fill item j up to c_j (so i cannot take it)
    zfull: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(n):
        for j in range(m):
            zfull[(i, j)] = mdl.addVar(vtype=GRB.BINARY, name=f"zfull[{i},{j}]")

    mdl.update()

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

    # zfull indicator constraints:
    # zfull[i,j] = 1  <=>  sum_{k!=i} x[k,j] >= c_j
    #
    # (A) If opponents already use >= c_j, then zfull must be 1.
    #     If zfull=0, forces sum_{k!=i} x[k,j] <= c_j-1.
    #
    # (B) If zfull=1, then opponents must be at least c_j.
    for i in range(n):
        for j in range(m):
            opp_use = gp.quicksum(x[(k, j)] for k in range(n) if k != i)
            cj = int(inst.c[j])

            # (A) full => zfull=1
            mdl.addConstr(
                opp_use <= (cj - 1) + (n - 1) * zfull[(i, j)],
                name=f"zfull_full_implies[{i},{j}]",
            )

            # (B) zfull=1 => full
            mdl.addConstr(
                opp_use >= cj * zfull[(i, j)],
                name=f"zfull_one_implies_full[{i},{j}]",
            )

    # Objective
    if objective == "social_welfare":
        obj = gp.quicksum(int(inst.p[i, j]) * x[(i, j)] for i in range(n) for j in range(m))
    else:
        raise ValueError(f"Unknown objective: {objective}")

    if sense.lower().startswith("max"):
        mdl.ModelSense = GRB.MAXIMIZE
        mdl.setObjective(obj)
    else:
        mdl.ModelSense = GRB.MINIMIZE
        mdl.setObjective(obj)

    return GKGModel(inst=inst, model=mdl, x=x, zfull=zfull)
