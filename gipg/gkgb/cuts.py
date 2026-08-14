"""Conditional Equilibrium Inequality (CEI) cuts for bilinear KPG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import gurobipy as gp
except Exception:  # pragma: no cover
    gp = None

import numpy as np

from .instance import KPGInstance


@dataclass(frozen=True)
class CEICut:
    player: int
    x_hat: np.ndarray      # shape (m,), binary deviation
    profit_hat: float       # deviation profit (evaluated at callback incumbent)
    alpha: float
    bigM: float


def build_cei_lhs_rhs(
    inst: KPGInstance,
    *,
    cut: CEICut,
    x_vars: Dict[Tuple[int, int], "gp.Var"],
    y_vars: Dict[Tuple[int, int, int], "gp.Var"],
    zfull_vars: Dict[Tuple[int, int], "gp.Var"],
) -> Tuple["gp.LinExpr", "gp.LinExpr"]:
    """Return (lhs, rhs) for an alpha-CEI with bilinear interaction terms.

    For bilinear KPG, the alpha-CEI for player i deviating to x_hat is:

        alpha * u_i(x)  >=  u_i(x_hat, x_{-i})  -  M * sum_{j: x_hat_j=1} zfull[i,j]

    Expanding:
        LHS = alpha * [sum_j p[i,j]*x[i,j] + sum_{k!=i} sum_j C[i,k,j]*y[i,k,j]]

        RHS = [sum_{j: x_hat_j=1} p[i,j]]                          (constant: direct profit)
            + [sum_{j: x_hat_j=1} sum_{k!=i} C[i,k,j] * x[k,j]]   (linear in x_{-i})
            - M * sum_{j: x_hat_j=1} zfull[i,j]                     (big-M relaxation)

    Both LHS and RHS are linear in the master model's variables (x, y, zfull).
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    i = int(cut.player)
    n, m = inst.n, inst.m
    alpha = float(cut.alpha)
    M = float(cut.bigM)

    # ---- LHS: alpha * u_i(x) ----
    lhs = gp.LinExpr()
    # Direct profits
    for j in range(m):
        lhs.addTerms(alpha * float(inst.p[i, j]), x_vars[(i, j)])
    # Interaction terms (using linearised y variables)
    for k in range(n):
        if k == i:
            continue
        for j in range(m):
            coeff = float(inst.C[i, k, j])
            if coeff != 0.0:
                lhs.addTerms(alpha * coeff, y_vars[(i, k, j)])

    # ---- RHS: deviation profit (depends on opponents) ----
    rhs = gp.LinExpr()

    # Constant part: direct profit of deviation
    const_profit = 0.0
    for j in range(m):
        if int(cut.x_hat[j]) == 1:
            const_profit += float(inst.p[i, j])
    rhs.addConstant(const_profit)

    # Variable part: interaction profit of deviation (linear in x_{-i})
    for j in range(m):
        if int(cut.x_hat[j]) == 1:
            for k in range(n):
                if k == i:
                    continue
                coeff = float(inst.C[i, k, j])
                if coeff != 0.0:
                    rhs.addTerms(coeff, x_vars[(k, j)])

    # Big-M relaxation
    for j in range(m):
        if int(cut.x_hat[j]) == 1:
            rhs.addTerms(-M, zfull_vars[(i, j)])

    return lhs, rhs
