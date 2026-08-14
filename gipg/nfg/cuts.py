"""Conditional Equilibrium Inequality (CEI) cuts for NFG.

CEI for GNEPs with Shapley cost sharing:
    For player i with deviation x_hat having deviation cost pi_hat,
    the CEI states:

        (current cost of i)  <=  alpha * (deviation cost)
                                  + M * Activation_B

    where Activation B is the per-state residual check (ISBP-style):
        For each edge e in supp(x_hat), for each y-state (ell, s) where
        c_e - (ell - s) < x_hat[e], the big-M penalty activates.

    This is SUFFICIENT for BR-produced deviations: since x_hat is a valid
    s-t flow (from the BR oracle), per-edge residual sufficiency implies
    flow-polytope feasibility (Proposition: flow-decomposition sufficiency).

    The current cost in y-variables:
        pi_i = sum_{e,ell>0,s>0}  w_e * s / ell  * y[i,e,ell,s]

Note: VEST cuts do NOT apply to NFG (flow polytope, not bounded simplex).
"""
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

from .instance import NFGInstance


@dataclass(frozen=True)
class CEICut:
    player: int
    x_hat: np.ndarray   # shape (n_edges,), integer deviation flow
    cost_hat: float      # deviation cost under current opponents
    alpha: float
    bigM: float

    def name(self) -> str:
        return f"CEI_p{self.player}_cost{self.cost_hat:.4g}_a{self.alpha:.6g}"


def _build_activation_B(
    inst: NFGInstance,
    *,
    cut: CEICut,
    y_vars: Dict[Tuple[int, int, int, int], "gp.Var"],
) -> "gp.LinExpr":
    """Build Activation B: per-state residual check.

    For each edge e in supp(x_hat), sum over y-states where the residual
    capacity c_e - (ell - s) is strictly less than x_hat[e].

    Returns M * sum_{...} y[i,e,ell,s].
    """
    i = int(cut.player)
    E = inst.n_edges
    M = float(cut.bigM)

    activation = gp.LinExpr()
    for e in range(E):
        x_hat_e = int(cut.x_hat[e])
        if x_hat_e <= 0:
            continue
        ce = int(inst.capacities[e])
        s_max_ie = inst.max_player_edge_flow(i, e)

        for ell in range(ce + 1):
            for s in range(min(ell, s_max_ie) + 1):
                key = (i, e, ell, s)
                if key not in y_vars:
                    continue
                opp_load = ell - s
                residual = ce - opp_load
                if residual < x_hat_e:
                    activation.addTerms(M, y_vars[key])

    return activation


def build_cei_lhs_rhs(
    inst: NFGInstance,
    *,
    cut: CEICut,
    y_vars: Dict[Tuple[int, int, int, int], "gp.Var"],
) -> Tuple["gp.LinExpr", "gp.LinExpr"]:
    """Build (lhs, rhs) for the alpha-CEI with Activation B (constant RHS).

    Constraint:  lhs <= rhs

    lhs = current cost of player i  (in terms of y variables)
          = sum_{e,ell>0,s>0}  w_e * s / ell * y[i,e,ell,s]
    rhs = alpha * cost_hat + Activation_B

    For minimisation alpha-approx:
        pi_i(x_i, x_{-i})  <=  alpha * pi_i(x_hat, x_{-i})
        with activation when deviation infeasible.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    i = int(cut.player)
    E = inst.n_edges
    alpha = float(cut.alpha)

    # ---- LHS: current cost of player i in y variables ----
    # pi_i = sum_e sum_{ell>0} sum_{s>0}  w_e * s / ell * y[i,e,ell,s]
    lhs = gp.LinExpr()
    for e in range(E):
        w_e = float(inst.edge_costs[e])
        ce = int(inst.capacities[e])
        for ell in range(1, ce + 1):
            s_max = min(ell, inst.max_player_edge_flow(i, e))
            for s in range(1, s_max + 1):
                key = (i, e, ell, s)
                if key in y_vars:
                    lhs.addTerms(w_e * float(s) / float(ell), y_vars[key])

    # ---- RHS: alpha * deviation cost + Activation B ----
    rhs = gp.LinExpr()
    rhs.addConstant(alpha * float(cut.cost_hat))

    # Activation B: per-state residual check
    activation = _build_activation_B(inst, cut=cut, y_vars=y_vars)
    rhs.add(activation)

    return lhs, rhs


def build_cei_with_opponent_load(
    inst: NFGInstance,
    *,
    cut: CEICut,
    y_vars: Dict[Tuple[int, int, int, int], "gp.Var"],
) -> Tuple["gp.LinExpr", "gp.LinExpr"]:
    """Alternative CEI that encodes the deviation cost in terms of opponent
    loads (via the y variables), rather than as a constant.

    This produces a *tighter* cut because the RHS also depends on the
    model variables (adapting to the opponents' configuration).

    Deviation cost on edge e:
        w_e * x_hat[e] / (opp_e + x_hat[e])
    where opp_e = ell - s (from y[i,e,ell,s]).

    For each y[i,e,ell,s]=1:
        opp_e = ell - s
        dev_cost_e = w_e * x_hat[e] / (ell - s + x_hat[e])
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    i = int(cut.player)
    E = inst.n_edges
    alpha = float(cut.alpha)

    # LHS: current cost (same as constant version)
    lhs = gp.LinExpr()
    for e in range(E):
        w_e = float(inst.edge_costs[e])
        ce = int(inst.capacities[e])
        for ell in range(1, ce + 1):
            s_max = min(ell, inst.max_player_edge_flow(i, e))
            for s in range(1, s_max + 1):
                key = (i, e, ell, s)
                if key in y_vars:
                    lhs.addTerms(w_e * float(s) / float(ell), y_vars[key])

    # RHS: alpha * deviation_cost(opponent_load) + Activation B
    rhs = gp.LinExpr()
    for e in range(E):
        x_hat_e = int(cut.x_hat[e])
        w_e = float(inst.edge_costs[e])
        ce = int(inst.capacities[e])
        s_max_ie = inst.max_player_edge_flow(i, e)

        for ell in range(ce + 1):
            for s in range(min(ell, s_max_ie) + 1):
                key = (i, e, ell, s)
                if key not in y_vars:
                    continue
                opp_e = ell - s
                if x_hat_e > 0:
                    total_dev = opp_e + x_hat_e
                    dev_cost_e = w_e * float(x_hat_e) / float(total_dev)
                else:
                    dev_cost_e = 0.0
                rhs.addTerms(alpha * dev_cost_e, y_vars[key])

    # Activation B: per-state residual check
    activation = _build_activation_B(inst, cut=cut, y_vars=y_vars)
    rhs.add(activation)

    return lhs, rhs
