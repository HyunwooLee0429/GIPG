"""Generalized knapsack games.

Each player selects items to maximize private profit subject to an individual
budget. Items carry per-item capacity limits shared across players, so one
player's selection can render another's infeasible.

Public entry points
-------------------
generate_instance
    Draw a random instance.
solve_gzr
    Generalized Zero-Regret: optimize a linear objective over the PNE set.
solve_abs
    alpha-Bisection Search: find the tightest alpha-approximate PNE.
solve_best_response
    Single-player best response against fixed opponents.
solve_social_optimum
    Maximum total welfare ignoring the equilibrium condition.
"""

from .instance import generate_gkg_instance as generate_instance
from .model import build_master_model
from .best_response import solve_best_response
from .heuristics import alpha_of_profile, brd_random_restart, random_feasible_profile
from .gzr import solve_gzr
from .abs_search import solve_abs as solve_abs, alpha_needed_for_profile
from .social_optimum import solve_social_optimum, compute_pos

__all__ = [
    "generate_instance",
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
