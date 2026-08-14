"""Heuristic algorithms for NFG: Random-Restart BRD and alpha computation."""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import time
import random

import numpy as np

from .instance import NFGInstance
from .objectives import player_cost, edge_loads, player_cost_against_opponents
from .best_response import solve_best_response, BRResult


# =====================================================================
# Random feasible profile generation
# =====================================================================

def random_feasible_profile(
    inst: NFGInstance,
    rng: random.Random,
    max_tries: int = 500,
) -> np.ndarray:
    """Generate a random feasible profile X (n_players x n_edges).

    Strategy: for each player, find a random s-t flow of value d_i
    using a simple path-based approach.
    """
    X = np.zeros((inst.n_players, inst.n_edges), dtype=int)
    rem_cap = inst.capacities.copy()

    for i in inst.players:
        d = int(inst.demands[i])
        flow = np.zeros(inst.n_edges, dtype=int)
        remaining = d

        for _ in range(max_tries):
            if remaining <= 0:
                break
            # Find an augmenting path from source to sink using BFS on residual graph
            path = _find_augmenting_path(
                inst, int(inst.sources[i]), int(inst.sinks[i]),
                rem_cap - flow.clip(0), rng,
            )
            if path is None:
                break
            # Find bottleneck capacity
            bottleneck = min(rem_cap[e] for e in path)
            bottleneck = min(bottleneck, remaining)
            if bottleneck <= 0:
                break
            # Push flow
            push = rng.randint(1, bottleneck)
            for e in path:
                flow[e] += push
            remaining -= push

        if remaining > 0:
            # Fallback: try to route remaining along any path
            for _ in range(max_tries):
                if remaining <= 0:
                    break
                path = _find_augmenting_path(
                    inst, int(inst.sources[i]), int(inst.sinks[i]),
                    rem_cap - flow.clip(0), rng,
                )
                if path is None:
                    break
                bottleneck = min(rem_cap[e] - flow[e] for e in path)
                bottleneck = min(bottleneck, remaining)
                if bottleneck <= 0:
                    break
                for e in path:
                    flow[e] += bottleneck
                remaining -= bottleneck

        X[i] = flow
        rem_cap -= flow

    return X


def _find_augmenting_path(
    inst: NFGInstance,
    source: int,
    sink: int,
    res_cap: np.ndarray,
    rng: random.Random,
) -> Optional[List[int]]:
    """BFS to find a path from source to sink with positive residual capacity.
    Returns list of edge indices, or None.
    """
    from collections import deque

    visited = {source}
    queue = deque([(source, [])])

    while queue:
        v, path = queue.popleft()
        if v == sink:
            return path

        # Randomise outgoing edge order for diversity
        out_edges = inst.outgoing(v)
        rng.shuffle(out_edges)

        for e in out_edges:
            w = inst.head(e)
            if w not in visited and int(res_cap[e]) > 0:
                visited.add(w)
                queue.append((w, path + [e]))

    return None


# =====================================================================
# Alpha computation
# =====================================================================

def alpha_of_profile(
    inst: NFGInstance,
    X: np.ndarray,
    *,
    br_time_limit: Optional[float] = None,
) -> float:
    """Compute alpha(X) = max_i  cost_i(X) / BR_cost_i(X_{-i}).

    For minimisation: alpha = max_i (current_cost / best_response_cost).
    alpha = 1 means exact PNE.  alpha > 1 means approximate.
    """
    loads = edge_loads(inst, X)
    ratios: List[float] = []

    for i in inst.players:
        curr = player_cost(inst, X, i)
        opp = loads - X[i]
        br = solve_best_response(inst, player=i, opp_load=opp, time_limit=br_time_limit)
        if br.cost is None or br.cost <= 0:
            if curr > 1e-9:
                return float("inf")
            ratios.append(1.0)
        else:
            ratios.append(curr / br.cost)

    return float(max(ratios)) if ratios else 1.0


# =====================================================================
# Random-Restart Best-Response Dynamics (RRR-BRD)
# =====================================================================

def brd_random_restart(
    inst: NFGInstance,
    *,
    max_init: int = 5,
    max_round: int = 20,
    seed: int = 0,
    br_time_limit: Optional[float] = None,
    end_round_profiles_out: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, bool, float]:
    """Random-restart BRD for NFG.

    Matches the paper's RRR-BRD description:
    - Runs up to `max_init` restarts (random initial profiles).
    - Each restart runs up to `max_round` rounds with random player permutations.
    - If BRD converges (no change in a round AND all BR solves succeeded),
      certify PNE and return immediately.
    - On the last restart, snapshot every end-of-round profile.
    - If no PNE is certified, evaluate alpha for every end-of-round profile
      from the last restart and return the one with minimum alpha.

    Returns (best_profile, pne_found, runtime).
    """
    t0 = time.time()
    rng = random.Random(seed)

    best_profile = np.zeros((inst.n_players, inst.n_edges), dtype=int)
    best_alpha = float("inf")
    pne_found = False

    # End-of-round profiles from the last restart
    last_restart_end_profiles: List[np.ndarray] = []

    for init_idx in range(max_init):
        is_last_restart = (init_idx == max_init - 1)

        X = random_feasible_profile(inst, rng)
        loads = edge_loads(inst, X)

        for _round in range(max_round):
            changed = False
            all_solved = True

            order = list(inst.players)
            rng.shuffle(order)

            for i in order:
                opp = loads - X[i]
                br = solve_best_response(
                    inst, player=i, opp_load=opp, time_limit=br_time_limit,
                )
                if br.cost is None:
                    all_solved = False
                    continue

                curr = player_cost(inst, X, i)
                tol = 1e-9
                if br.cost < curr - tol:
                    # Update player i's strategy
                    old_flow = X[i].copy()
                    X[i] = br.x_hat
                    loads = loads - old_flow + br.x_hat
                    changed = True

            # Snapshot end-of-round profile on last restart
            if is_last_restart:
                snap = X.copy()
                last_restart_end_profiles.append(snap)
                if end_round_profiles_out is not None:
                    end_round_profiles_out.append(X.copy())

            # Certify PNE only if nothing changed AND all BRs were solved
            if (not changed) and all_solved:
                pne_found = True
                return X.copy(), True, time.time() - t0

        # No certified PNE in this restart
        best_profile = X.copy()

    # No certified PNE across restarts:
    # Select best alpha among end-of-round profiles from last restart
    if last_restart_end_profiles:
        best_a = float("inf")
        best_x = last_restart_end_profiles[-1]
        for prof in last_restart_end_profiles:
            a = alpha_of_profile(inst, prof, br_time_limit=br_time_limit)
            if a < best_a:
                best_a = a
                best_x = prof
        best_profile = best_x

    return best_profile, pne_found, time.time() - t0
