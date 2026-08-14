from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:  # pragma: no cover
    gp = None
    GRB = None

# Shared quiet Gurobi environment (created once, suppresses param-change messages)
_quiet_env: Optional["gp.Env"] = None

def _get_quiet_env() -> "gp.Env":
    global _quiet_env
    if _quiet_env is None and gp is not None:
        _quiet_env = gp.Env(empty=True)
        _quiet_env.setParam("OutputFlag", 0)
        _quiet_env.start()
    return _quiet_env

def best_response_linearized(
    player: int,
    players: List[int],
    bins: List[int],
    costs: Dict[int, float],
    capacities: Dict[int, int],
    weights: Dict[int, int],
    x_current: Dict[int, Dict[int, int]],
    time_limit: Optional[float] = None,
    threads: Optional[int] = None,
) -> Tuple[Dict[int, int], Optional[float]]:
    """
    Linearized best-response model with binaries delta[j,l] choosing
    how much load l the player puts in bin j.

    Objective coefficient uses the *current opponents' load* in that bin:
        gamma[j,l] = c_j * l / (occ_other[j] + l)  (0 if denominator is 0)
    """
    if gp is None:
        raise ImportError("gurobipy is required for best_response_linearized()")

    occ_other = {j: sum(x_current[i][j] for i in players if i != player) for j in bins}
    rem = {j: capacities[j] - occ_other[j] for j in bins}
    max_assign = {j: min(rem[j], weights[player]) for j in bins}

    model = gp.Model(f"BR_player_{player}", env=_get_quiet_env())
    model.Params.Threads = 16 if threads is None else int(threads)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    delta = {}
    for j in bins:
        for l in range(max_assign[j] + 1):
            delta[j, l] = model.addVar(vtype=GRB.BINARY, name=f"delta[{j},{l}]")

    # each bin: choose exactly one l
    for j in bins:
        model.addConstr(gp.quicksum(delta[j, l] for l in range(max_assign[j] + 1)) == 1)

    # total load equals weight
    model.addConstr(
        gp.quicksum(l * delta[j, l] for j in bins for l in range(max_assign[j] + 1)) == weights[player]
    )

    obj = gp.quicksum(
        (costs[j] * l / (occ_other[j] + l) if (occ_other[j] + l) > 0 else 0.0) * delta[j, l]
        for j in bins for l in range(max_assign[j] + 1)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # warm-start from current allocation if feasible
    for j in bins:
        l0 = x_current[player][j]
        if 0 <= l0 <= max_assign[j]:
            delta[j, l0].Start = 1

    model.optimize()

    alloc = {j: 0 for j in bins}
    if model.Status == GRB.OPTIMAL:
        for j in bins:
            for l in range(max_assign[j] + 1):
                if delta[j, l].X > 0.5:
                    alloc[j] = int(l)
                    break
        return alloc, float(model.ObjVal)
    return alloc, None


class BestResponseOracle:
    """Reusable best-response IP for a single player.

    This is designed for use inside a Gurobi callback.

    We build one model per player with variables v[j,l] for l=0..min(u_j, w_i).
    At each call, we:
      (i) update the residual-capacity RHS per bin using the current opponents load,
      (ii) update objective coefficients r_{j,l} = c_j * l / (opp_j + l),
      (iii) optimize and return the allocation.

    This avoids rebuilding the BR model at every incumbent.
    """

    def __init__(self, inst, player: int, *, threads: Optional[int] = None, log_to_console: int = 0):
        if gp is None:
            raise ImportError("gurobipy is required")

        self.inst = inst
        self.player = int(player)
        self.players = list(inst.players)
        self.bins = list(inst.bins)
        self.costs = inst.costs
        self.capacities = inst.capacities
        self.w_i = int(inst.weights[self.player])

        self.model = gp.Model(f"BR_oracle_p{self.player}", env=_get_quiet_env())
        self.model.Params.Threads = 16 if threads is None else int(threads)
        if log_to_console:
            self.model.Params.LogToConsole = log_to_console

        # v[j,l]
        self.v = {}
        self.max_l = {}
        for j in self.bins:
            ml = min(int(self.capacities[j]), self.w_i)
            self.max_l[j] = ml
            for l in range(ml + 1):
                self.v[(j, l)] = self.model.addVar(vtype=GRB.BINARY, name=f"v[{j},{l}]")

        # choose exactly one l per bin
        for j in self.bins:
            self.model.addConstr(gp.quicksum(self.v[(j, l)] for l in range(self.max_l[j] + 1)) == 1)

        # total weight
        self.model.addConstr(
            gp.quicksum(l * self.v[(j, l)] for j in self.bins for l in range(self.max_l[j] + 1)) == self.w_i
        )

        # residual capacity constraints per bin: sum l v[j,l] <= cap_rhs[j]
        # RHS will be updated per call.
        self.cap_constr = {}
        for j in self.bins:
            expr = gp.quicksum(l * self.v[(j, l)] for l in range(self.max_l[j] + 1))
            self.cap_constr[j] = self.model.addConstr(expr <= int(self.capacities[j]), name=f"cap[{j}]")

        # Objective coefficients are set per call by update_objective().
        self.model.setObjective(0.0, GRB.MINIMIZE)
        self.model.update()

    def solve(
        self,
        opponents_load: Dict[int, int],
        *,
        time_limit: Optional[float] = None,
    ) -> Tuple[Dict[int, int], Optional[float]]:
        """Solve the BR against fixed opponents_load[j]."""
        if time_limit is not None:
            self.model.Params.TimeLimit = float(time_limit)

        # Update residual capacities
        for j in self.bins:
            cap = int(self.capacities[j]) - int(opponents_load.get(j, 0))
            if cap < 0:
                cap = 0
            self.cap_constr[j].RHS = cap

        # Update objective coefficients
        obj = gp.LinExpr()
        for j in self.bins:
            opp = float(opponents_load.get(j, 0))
            cj = float(self.costs[j])
            for l in range(self.max_l[j] + 1):
                if l == 0:
                    coef = 0.0
                else:
                    denom = opp + float(l)
                    coef = (cj * float(l) / denom) if denom > 0 else 0.0
                obj.addTerms(coef, self.v[(j, l)])
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()

        self.model.optimize()

        alloc = {j: 0 for j in self.bins}
        if self.model.Status == GRB.OPTIMAL:
            for j in self.bins:
                for l in range(self.max_l[j] + 1):
                    if self.v[(j, l)].X > 0.5:
                        alloc[j] = int(l)
                        break
            return alloc, float(self.model.ObjVal)
        return alloc, None
