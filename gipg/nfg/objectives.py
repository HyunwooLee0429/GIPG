"""Cost functions and player cost evaluation for Network Formation Games.

Shapley / proportional cost sharing:
    Player i's cost on edge e = w_e * x_{i,e} / ell_e
    where w_e is the fixed edge cost and ell_e = sum_k x_{k,e}.

    Total player cost:  pi_i(X) = sum_e  w_e * X[i,e] / ell_e
"""
from __future__ import annotations

from typing import List
import numpy as np

from .instance import NFGInstance


# =====================================================================
# Edge-level cost sharing
# =====================================================================

def edge_cost_share(w_e: float, s: int, ell: int) -> float:
    """Shapley cost share for a player sending s units on edge e with total load ell.

    Returns w_e * s / ell  if ell > 0 and s > 0, else 0.
    """
    if ell <= 0 or s <= 0:
        return 0.0
    return float(w_e) * float(s) / float(ell)


# =====================================================================
# Profile-level computations
# =====================================================================

def edge_loads(inst: NFGInstance, X: np.ndarray) -> np.ndarray:
    """Compute total load on each edge.  X shape (n_players, n_edges)."""
    return X.sum(axis=0)


def player_cost(inst: NFGInstance, X: np.ndarray, player: int) -> float:
    """Cost of player i under profile X.

    pi_i(X) = sum_e  w_e * X[i,e] / ell_e
    """
    loads = edge_loads(inst, X)
    i = int(player)
    cost = 0.0
    for e in inst.edge_indices:
        s = int(X[i, e])
        ell = int(loads[e])
        if s > 0 and ell > 0:
            cost += float(inst.edge_costs[e]) * float(s) / float(ell)
    return cost


def all_player_costs(inst: NFGInstance, X: np.ndarray) -> np.ndarray:
    """Return array of player costs under profile X."""
    return np.array([player_cost(inst, X, i) for i in inst.players])


def social_cost(inst: NFGInstance, X: np.ndarray) -> float:
    """Total social cost = sum of all player costs.

    Note: under Shapley sharing, the social cost equals sum_e w_e * 1_{ell_e > 0},
    i.e. the total cost of all edges used.  This is because:
        sum_i w_e * x_{i,e} / ell_e = w_e * ell_e / ell_e = w_e  for each used edge.
    """
    loads = edge_loads(inst, X)
    cost = 0.0
    for e in inst.edge_indices:
        if int(loads[e]) > 0:
            cost += float(inst.edge_costs[e])
    return cost


def player_cost_against_opponents(
    inst: NFGInstance,
    player: int,
    xi: np.ndarray,
    opp_load: np.ndarray,
) -> float:
    """Cost of player i playing xi given opponent edge loads opp_load.

    pi_i = sum_e  w_e * xi[e] / (opp_load[e] + xi[e])
    """
    cost = 0.0
    for e in inst.edge_indices:
        s = int(xi[e])
        if s > 0:
            total = int(opp_load[e]) + s
            cost += float(inst.edge_costs[e]) * float(s) / float(total)
    return cost
