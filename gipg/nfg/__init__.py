"""Network formation games with integer-splittable routing.

Each player routes an integer demand from a common source to a common sink over
a shared network. An edge's cost is split among its users in proportion to the
flow each sends, and edge capacities are shared, so both the objective and the
feasible sets couple the players.

Public entry points
-------------------
generate_instance
    Draw an instance from the capacity x symmetry design used in the paper;
    pass ``graph_type="layered"`` for the layered networks.
solve_gzr
    Generalized Zero-Regret: optimize a linear objective over the PNE set.
solve_abs
    alpha-Bisection Search: find the tightest alpha-approximate PNE.
solve_best_response
    Single-player best response against fixed opponents.
solve_social_optimum
    Minimum total cost ignoring the equilibrium condition.
"""

from .instance import (
    generate_nfg_2x2 as generate_instance,
    generate_nfg_instance as generate_random_instance,
    save_instance,
    load_instance,
)
from .objectives import (
    edge_loads,
    edge_cost_share,
    player_cost,
    all_player_costs,
    social_cost,
)
from .model import build_master_model
from .best_response import solve_best_response
from .heuristics import alpha_of_profile, brd_random_restart, random_feasible_profile
from .gzr import solve_gzr
from .abs_search import solve_abs as solve_abs
from .social_optimum import solve_social_optimum, compute_pos

__all__ = [
    "generate_instance",
    "generate_random_instance",
    "save_instance",
    "load_instance",
    "edge_loads",
    "edge_cost_share",
    "player_cost",
    "all_player_costs",
    "social_cost",
    "build_master_model",
    "solve_best_response",
    "alpha_of_profile",
    "brd_random_restart",
    "random_feasible_profile",
    "solve_gzr",
    "solve_abs",
    "solve_social_optimum",
    "compute_pos",
]
