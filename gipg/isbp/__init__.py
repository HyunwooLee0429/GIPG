"""Integer splittable bin packing games.

Players place their integer demand across shared bins. A bin's cost is split
among its occupants in proportion to the load each places there, so a player's
cost depends on the whole profile and the shared bin capacities couple the
players' feasible sets.

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
    Minimum total cost ignoring the equilibrium condition.
"""

from .instance import (
    generate_isbp_instance as generate_instance,
    identical_bin_groups,
    bin_to_group_map,
)
from .objectives import (
    profile_costs,
    total_loads_from_profile,
    player_cost_from_profile,
)
from .best_response import best_response_linearized as solve_best_response
from .heuristics import (
    alpha_of_profile,
    brd_random_restart,
    is_regular_instance,
    random_feasible_profile,
    run_all_heuristics,
)
from .gzr import (
    build_full_discretized_model,
    solve_gzr as solve_gzr,
)
from .abs_search import solve_abs
from .social_optimum import solve_social_optimum, compute_pos

__all__ = [
    "generate_instance",
    "identical_bin_groups",
    "bin_to_group_map",
    "profile_costs",
    "total_loads_from_profile",
    "player_cost_from_profile",
    "solve_best_response",
    "alpha_of_profile",
    "brd_random_restart",
    "is_regular_instance",
    "random_feasible_profile",
    "run_all_heuristics",
    "build_full_discretized_model",
    "solve_gzr",
    "solve_abs",
    "solve_social_optimum",
    "compute_pos",
]
