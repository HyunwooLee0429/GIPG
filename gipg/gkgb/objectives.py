"""Utility computation for bilinear KPG."""
from __future__ import annotations

import numpy as np

from .instance import KPGInstance


def player_utility(inst: KPGInstance, X: np.ndarray, player: int) -> float:
    """Compute u_i(X) = direct profits + interaction terms.

    u_i = sum_j p[i,j]*X[i,j] + sum_{k!=i} sum_j C[i,k,j]*X[i,j]*X[k,j]
    """
    i = player
    n, m = inst.n, inst.m
    util = float(np.dot(inst.p[i], X[i]))
    for k in range(n):
        if k == i:
            continue
        util += float(np.dot(inst.C[i, k] * X[i], X[k]))
    return util


def all_player_utilities(inst: KPGInstance, X: np.ndarray) -> np.ndarray:
    """Compute utilities for all players.  Returns shape (n,) array."""
    return np.array([player_utility(inst, X, i) for i in range(inst.n)])


def social_welfare(inst: KPGInstance, X: np.ndarray) -> float:
    """Total social welfare = sum_i u_i(X)."""
    return float(sum(player_utility(inst, X, i) for i in range(inst.n)))
