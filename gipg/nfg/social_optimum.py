"""Social optimum and Price of Stability (POS) computation for NFG.

The social optimum is the minimum social cost over all feasible strategy
profiles (ignoring equilibrium constraints).  It is computed by solving
the master MIP without any CEI lazy cuts.

POS = social_cost(best equilibrium) / social_optimum_cost
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np

from .instance import NFGInstance
from .model import build_master_model
from .objectives import social_cost


@dataclass
class SocialOptResult:
    """Result of a social optimum computation."""
    status: str
    runtime: float
    opt_cost: Optional[float]      # optimal social cost (None if infeasible)
    profile: Optional[np.ndarray]  # optimal profile X (None if infeasible)


def solve_social_optimum(
    inst: NFGInstance,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> SocialOptResult:
    """Compute the social optimum: min social_cost(X) s.t. X feasible.

    This solves the master MIP (same constraints as GZR) but without
    any CEI lazy cuts, yielding the globally optimal feasible profile.

    Parameters
    ----------
    inst : NFGInstance
        Game instance.
    time_limit : float, optional
        Solver time limit in seconds.
    mip_gap : float, optional
        Relative MIP gap tolerance.
    threads : int, optional
        Number of solver threads.
    verbose : bool
        If True, print solver output.

    Returns
    -------
    SocialOptResult
        Contains optimal social cost, profile, and solver status.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    start = time.time()

    # Build the master model (same feasibility constraints + social cost objective)
    nfgm = build_master_model(
        inst,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        verbose=verbose,
        add_vest_cut=False,  # SO is not a PNE; weaker VEST cuts do not apply
    )
    mdl = nfgm.model

    # Disable lazy constraints (not needed for social optimum)
    mdl.Params.LazyConstraints = 0

    # Solve without callback (pure MIP, no equilibrium constraints)
    mdl.optimize()

    runtime = time.time() - start

    # Extract result
    profile = None
    opt_cost = None

    if mdl.SolCount > 0:
        profile = nfgm.extract_profile()
        opt_cost = social_cost(inst, profile)
        status = "OPTIMAL" if mdl.Status == GRB.OPTIMAL else "FEASIBLE"
    elif mdl.Status == GRB.INFEASIBLE:
        status = "INFEASIBLE"
    elif mdl.Status == GRB.TIME_LIMIT:
        status = "TIME_LIMIT"
    else:
        status = f"STATUS_{mdl.Status}"

    return SocialOptResult(
        status=status,
        runtime=runtime,
        opt_cost=opt_cost,
        profile=profile,
    )


def compute_pos(
    equilibrium_cost: float,
    social_opt_cost: float,
) -> float:
    """Compute the Price of Stability (POS).

    POS = social_cost(best equilibrium) / social_optimum_cost

    Parameters
    ----------
    equilibrium_cost : float
        Social cost at the best Nash equilibrium found.
    social_opt_cost : float
        Social cost at the socially optimal feasible profile.

    Returns
    -------
    float
        The POS ratio (>= 1.0 by definition).
    """
    if social_opt_cost <= 0:
        return float("inf")
    return equilibrium_cost / social_opt_cost
