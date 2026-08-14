"""Social optimum solver for ISBP (minimise total cost without equilibrium constraints)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

from .instance import ISBPInstance
from .gzr import build_full_discretized_model


@dataclass
class SOResult:
    status: str
    opt_cost: Optional[float]   # total cost at social optimum (lower = better)
    runtime: float
    profile: Optional[Dict[int, Dict[int, int]]]


def solve_social_optimum(
    inst: ISBPInstance,
    *,
    time_limit: float = 600.0,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> SOResult:
    """Solve the social optimum: min total cost WITHOUT equilibrium (EI) constraints.

    This is the cooperative benchmark — all players coordinated to minimise
    total cost subject to weight and capacity constraints only.
    """
    model, x, y, z, groups = build_full_discretized_model(
        inst,
        add_bin_symmetry_breaking=True,
        add_vest_cut=False,  # SO is not a PNE; VEST cuts do not apply
        add_player_anchoring=False,
        log_to_console=1 if verbose else 0,
    )

    # Disable lazy constraints (not needed for SO)
    model.Params.LazyConstraints = 0
    model.Params.Threads = 16 if threads is None else int(threads)
    model.Params.TimeLimit = time_limit

    t0 = time.time()
    model.optimize()  # no callback — pure optimisation
    runtime = time.time() - t0

    if model.Status == GRB.OPTIMAL:
        status = "OPTIMAL"
    elif model.Status == GRB.TIME_LIMIT:
        status = "TIME_LIMIT"
    elif model.Status == GRB.INFEASIBLE:
        status = "INFEASIBLE"
    else:
        status = f"STATUS_{model.Status}"

    profile = None
    opt_cost = None
    if model.SolCount > 0:
        # Extract profile from x variables
        x_prof = {i: {j: 0 for j in inst.bins} for i in inst.players}
        for i in inst.players:
            for j in inst.bins:
                s_max = min(int(inst.capacities[j]), int(inst.weights[i]))
                for s in range(s_max + 1):
                    if x[(i, j, s)].X > 0.5:
                        x_prof[i][j] = int(s)
                        break
        profile = x_prof
        opt_cost = float(model.ObjVal)

    return SOResult(
        status=status,
        opt_cost=opt_cost,
        runtime=float(runtime),
        profile=profile,
    )


def compute_pos(so_cost: float, best_pne_cost: float) -> float:
    """Price of Stability for minimisation games.

    POS = best_PNE_cost / SO_cost >= 1.  Closer to 1 = less inefficiency.
    """
    if so_cost <= 0:
        return float("inf")
    return best_pne_cost / so_cost
