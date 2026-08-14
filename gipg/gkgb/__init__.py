"""Generalized knapsack games with reciprocally bilinear payoffs.

Extends the generalized knapsack game with interaction terms: a player's profit
from an item depends bilinearly on the opponents' selections of that item. The
products are handled by McCormick linearization in the master model.

Public entry points
-------------------
load_instance
    Read an instance from the bundled benchmark set.
solve_gzr
    Generalized Zero-Regret: optimize a linear objective over the PNE set.
solve_abs
    alpha-Bisection Search: find the tightest alpha-approximate PNE.
solve_best_response
    Single-player best response against fixed opponents.
solve_social_optimum
    Maximum total welfare ignoring the equilibrium condition.
"""

from .instance import (
    load_dragotto_instance as load_instance,
    enumerate_dragotto_instances,
    enumerate_lee_instances,
)
from .objectives import player_utility, all_player_utilities, social_welfare
from .model import build_master_model
from .best_response import solve_best_response
from .heuristics import alpha_of_profile, brd_random_restart, random_feasible_profile
from .gzr import solve_gzr
from .abs_search import solve_abs as solve_abs, alpha_needed_for_profile
from .social_optimum import solve_social_optimum, compute_pos

__all__ = [
    "load_instance",
    "enumerate_dragotto_instances",
    "enumerate_lee_instances",
    "player_utility",
    "all_player_utilities",
    "social_welfare",
    "build_master_model",
    "solve_best_response",
    "alpha_of_profile",
    "brd_random_restart",
    "random_feasible_profile",
    "solve_gzr",
    "solve_abs",
    "alpha_needed_for_profile",
    "solve_social_optimum",
    "compute_pos",
]
