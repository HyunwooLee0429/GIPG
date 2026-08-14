from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any, Optional
import numpy as np

@dataclass(frozen=True)
class ISBPInstance:
    players: List[int]
    bins: List[int]
    costs: Dict[int, float]
    capacities: Dict[int, int]
    weights: Dict[int, int]
    T: Dict[int, List[int]]
    S: Dict[int, Dict[int, Dict[int, List[int]]]]

def generate_isbp_instance(
    n_players: int,
    m_bins: int,
    c_j_range: int | Tuple[int, int],
    u_j_range: int | Tuple[int, int],
    w_i_range: int | Tuple[int, int],
    seed: Optional[int] = None,
) -> ISBPInstance:
    """
    Generate a random (or constant-parameter) ISBP instance and the auxiliary sets T and S.

    - players i in {0,...,n-1}
    - bins j in {0,...,m-1}
    - capacity u_j, cost c_j
    - player weight w_i

    T[j] = {0,...,u_j}
    S[i][j][t] = {0,..., min(t, w_i)}
    """
    rng = np.random.default_rng(seed)
    players = list(range(n_players))
    bins = list(range(m_bins))

    # costs
    if isinstance(c_j_range, int):
        costs = {j: float(c_j_range) for j in bins}
    else:
        lo, hi = c_j_range
        costs = {j: float(rng.integers(lo, hi)) for j in bins}

    # capacities
    if isinstance(u_j_range, int):
        capacities = {j: int(u_j_range) for j in bins}
    else:
        lo, hi = u_j_range
        capacities = {j: int(rng.integers(lo, hi)) for j in bins}

    # weights
    if isinstance(w_i_range, int):
        weights = {i: int(w_i_range) for i in players}
    else:
        lo, hi = w_i_range
        weights = {i: int(rng.integers(lo, hi)) for i in players}

    T = {j: list(range(0, capacities[j] + 1)) for j in bins}
    S = {
        i: {
            j: {
                t: list(range(0, min(t, weights[i]) + 1))
                for t in T[j]
            }
            for j in bins
        }
        for i in players
    }
    return ISBPInstance(players, bins, costs, capacities, weights, T, S)


from collections import defaultdict

def identical_bin_groups(inst: ISBPInstance) -> List[List[int]]:
    """Return groups of identical bins based on (capacity, cost).

    Two bins j,k are considered identical if (u_j, c_j) matches exactly.
    The output is a list of groups (each a list of bin indices). Groups of size 1
    are included for convenience.
    """
    groups: Dict[Tuple[int, float], List[int]] = defaultdict(list)
    for j in inst.bins:
        key = (int(inst.capacities[j]), float(inst.costs[j]))
        groups[key].append(j)
    # sort bins within each group for determinism
    out = [sorted(js) for js in groups.values()]
    # sort groups by key for determinism
    out.sort(key=lambda g: (inst.capacities[g[0]], inst.costs[g[0]], len(g), g[0]))
    return out

def bin_to_group_map(groups: List[List[int]]) -> Dict[int, int]:
    """Map bin j -> group index in `groups`."""
    m: Dict[int, int] = {}
    for gi, g in enumerate(groups):
        for j in g:
            m[j] = gi
    return m
