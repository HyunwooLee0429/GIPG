from __future__ import annotations

from typing import Dict, Tuple, List, Iterable, Iterator, Optional
import itertools
import random

try:
    import gurobipy as gp
except Exception:  # pragma: no cover
    gp = None

from .instance import ISBPInstance


def build_equilibrium_inequality_expr(
    *,
    player: int,
    BR: Dict[int, int],
    alpha: float,
    bigM: float,
    y: Dict[Tuple[int, int, int, int], "gp.Var"],
    costs: Dict[int, float],
    capacities: Dict[int, int],
    T: Dict[int, List[int]],
    S: Dict[int, Dict[int, Dict[int, List[int]]]],
    bins: List[int],
) -> Tuple["gp.LinExpr", "gp.LinExpr"]:
    """Return (lhs, rhs) LinExpr for the code-consistent ISBP equilibrium inequality.

    LHS is player i's current cost:  sum_{j,t,s} c_j * (s/t) * y[i,j,t,s].

    RHS evaluates the deviation BR against the incumbent opponents' load.
    For each incumbent lifted state (j,t,s), opponents load is (t-s) and
    post-deviation total load is (t-s) + hat_s_j.

    The big-M penalty activates for states where the deviation load is infeasible:
        u_j - (t-s) < hat_s_j  <=>  u_j - t + s < hat_s_j.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    i = int(player)
    best_bins = [j for j, s_hat in BR.items() if int(s_hat) > 0]

    lhs = gp.LinExpr()
    for j in bins:
        cj = float(costs[j])
        for t in T[j]:
            if t == 0:
                continue
            for s in S[i][j][t]:
                lhs.addTerms(cj * (float(s) / float(t)), y[(i, j, t, s)])

    rhs = gp.LinExpr()
    # Deviation-cost part (scaled by alpha)
    for j in best_bins:
        s_hat = float(BR[j])
        cj = float(costs[j])
        for t in T[j]:
            for s in S[i][j][t]:
                denom = float(t - s) + s_hat
                coef = (cj * s_hat / denom) if denom > 0 else 0.0
                rhs.addTerms(alpha * coef, y[(i, j, t, s)])

    # Big-M infeasibility guard (NOT scaled by alpha)
    for j in best_bins:
        s_hat = int(BR[j])
        uj = int(capacities[j])
        for t in T[j]:
            for s in S[i][j][t]:
                if uj - t + s < s_hat:
                    rhs.addTerms(float(bigM), y[(i, j, t, s)])

    return lhs, rhs


def deviation_cost_from_opponents_load(
    inst: ISBPInstance,
    player: int,
    BR: Dict[int, int],
    opponents_load: Dict[int, int],
) -> float:
    """Compute the deviation cost for a fixed BR allocation under fixed opponents load."""
    i = int(player)
    total = 0.0
    for j in inst.bins:
        s = int(BR.get(j, 0))
        if s <= 0:
            continue
        opp = float(opponents_load.get(j, 0))
        denom = opp + float(s)
        if denom <= 0:
            continue
        total += float(inst.costs[j]) * float(s) / denom
    return float(total)


def _sample_combinations(items: List[int], k: int, max_count: int, sample_fraction: float, rng: random.Random) -> List[Tuple[int, ...]]:
    """Sample up to max_count combinations of size k."""
    all_count = 0
    try:
        # Python 3.8+: comb may not exist; fallback to iter.
        from math import comb

        all_count = comb(len(items), k)
    except Exception:
        all_count = 0

    # target number of combos
    if all_count > 0:
        target = max(1, int((sample_fraction * all_count) + 0.999))
    else:
        target = max(1, int(sample_fraction * 100))
    target = min(target, max_count)

    combos = list(itertools.combinations(items, k))
    if len(combos) <= target:
        return combos
    return rng.sample(combos, target)


def _unique_permutations(multiset: List[int], max_count: int, rng: random.Random) -> List[Tuple[int, ...]]:
    """Return up to max_count distinct permutations of a multiset."""
    # If k is small, enumerating all unique permutations is fine.
    k = len(multiset)
    if k <= 8:
        perms = list(set(itertools.permutations(multiset)))
        if len(perms) <= max_count:
            return perms
        return rng.sample(perms, max_count)

    # Otherwise, sample by random shuffling and de-dup.
    target = max_count
    seen = set()
    base = list(multiset)
    tries = 0
    while len(seen) < target and tries < 2000:
        rng.shuffle(base)
        seen.add(tuple(base))
        tries += 1
    return list(seen)


def symmetric_br_patterns(
    *,
    inst: ISBPInstance,
    br_alloc: Dict[int, int],
    groups: List[List[int]],
    sample_fraction: float = 0.50,
    max_combinations: int = 50,
    max_permutations: int = 50,
    rng_seed: int = 0,
) -> Iterator[Dict[int, int]]:
    """
    Generate symmetric BR load patterns by permuting the positive loads of `br_alloc`
    across identical bins (within a single identical-bin group).

    Notes:
      - The generated patterns are intended for *globally valid* EIs; they are NOT required
        to be best responses for the current incumbent profile.
      - We only generate patterns when the positive support of `br_alloc` lies entirely
        inside one identical-bin group. (This matches the theoretical exposition and
        avoids cross-group combinatorial explosion.)

    Yields:
        dict j -> s'_j (includes zeros for bins not used).
    """
    rng = random.Random(rng_seed)

    pos_bins = [j for j, s in br_alloc.items() if int(s) > 0]
    k = len(pos_bins)
    if k <= 1:
        return

    # Identify a group D that contains all pos_bins
    D: Optional[List[int]] = None
    for g in groups:
        if all(j in g for j in pos_bins):
            D = g
            break
    if D is None or len(D) < k:
        return

    loads = [int(br_alloc[j]) for j in pos_bins]
    combos = _sample_combinations(D, k, max_combinations, sample_fraction, rng)
    perms = _unique_permutations(loads, max_permutations, rng)

    for C in combos:
        for pi in perms:
            br2 = {j: 0 for j in inst.bins}
            for idx, j in enumerate(C):
                br2[j] = int(pi[idx])
            yield br2
