from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import gurobipy as gp
except Exception:  # pragma: no cover
    gp = None

import numpy as np

from .instance import GKGInstance


@dataclass(frozen=True)
class CEICut:
    player: int
    x_hat: np.ndarray  # shape (m,)
    profit_hat: int
    alpha: float
    bigM: float

    def name(self) -> str:
        return f"CEI_p{self.player}_profit{self.profit_hat}_a{self.alpha:.6g}"


def build_cei_lhs_rhs(
    inst: GKGInstance,
    *,
    cut: CEICut,
    x_vars: Dict[Tuple[int, int], "gp.Var"],
    zfull_vars: Dict[Tuple[int, int], "gp.Var"],
) -> tuple["gp.LinExpr", "gp.LinExpr"]:
    """Return (lhs, rhs) for alpha-CEI with zfull activation.

    Maximization alpha-approx notion implemented as:
        alpha * pi_i(x_i) >= pi_i(x_hat) - M * sum_{j: x_hat_j=1} zfull_{i,j}

    where zfull_{i,j}=1 indicates item j is full without i, hence deviation infeasible.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    i = int(cut.player)
    m = inst.m
    alpha = float(cut.alpha)
    M = float(cut.bigM)

    lhs = gp.LinExpr()
    for j in range(m):
        lhs.addTerms(alpha * float(inst.p[i, j]), x_vars[(i, j)])

    rhs = gp.LinExpr()
    rhs.addConstant(float(cut.profit_hat))

    # activation term
    for j in range(m):
        if int(cut.x_hat[j]) == 1:
            rhs.addTerms(-M, zfull_vars[(i, j)])

    return lhs, rhs
