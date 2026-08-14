from __future__ import annotations

from typing import Dict, Tuple, List


def profile_costs(
    players: List[int],
    bins: List[int],
    costs: Dict[int, float],
    x_profile: Dict[int, Dict[int, int]],
) -> Tuple[Dict[int, float], float]:
    """
    Given a profile x_profile[player][bin] = load assigned by player to bin,
    compute each player's cost and total cost.

    Cost sharing:
      unit_cost[j] = c_j / total_load_in_bin_j (0 if empty)
      cost_i = sum_j unit_cost[j] * x_{i,j}
    """
    unit_cost: Dict[int, float] = {}
    for j in bins:
        tot = sum(int(x_profile[i][j]) for i in players)
        unit_cost[j] = (float(costs[j]) / float(tot)) if tot != 0 else 0.0

    selfish = {i: sum(unit_cost[j] * float(x_profile[i][j]) for j in bins) for i in players}
    total = float(sum(selfish.values()))
    return selfish, total


def total_loads_from_profile(
    players: List[int],
    bins: List[int],
    x_profile: Dict[int, Dict[int, int]],
) -> Dict[int, int]:
    """Return total load per bin under a profile."""
    return {j: sum(int(x_profile[i][j]) for i in players) for j in bins}


def player_cost_from_profile(
    *,
    players: List[int],
    bins: List[int],
    costs: Dict[int, float],
    x_profile: Dict[int, Dict[int, int]],
    player: int,
) -> float:
    """Compute one player's cost under the cost-sharing rule."""
    loads = total_loads_from_profile(players, bins, x_profile)
    i = int(player)
    cost = 0.0
    for j in bins:
        t = loads[j]
        s = int(x_profile[i][j])
        if t > 0 and s > 0:
            cost += float(costs[j]) * (float(s) / float(t))
    return float(cost)
