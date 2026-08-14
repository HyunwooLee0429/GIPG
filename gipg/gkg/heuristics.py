"""Heuristic algorithms for GKG: Random-Restart BRD and alpha computation."""
from __future__ import annotations

from typing import List, Tuple, Optional
import time
import random

import numpy as np

from .instance import GKGInstance
from .best_response import solve_best_response


# =====================================================================
# Alpha computation
# =====================================================================

def alpha_of_profile(
    inst: GKGInstance,
    X: np.ndarray,
    *,
    br_time_limit: Optional[float] = None,
) -> float:
    """Compute alpha(X) = max_i  pi_BR(X_{-i}) / pi_i(X).

    For maximisation: alpha = max_i (BR_profit / current_profit).
    alpha = 1 means exact PNE.  alpha > 1 means approximate.
    """
    n, m = inst.n, inst.m
    ratios: List[float] = []

    for i in range(n):
        pi_cur = float(sum(int(inst.p[i, j]) * int(X[i, j]) for j in range(m)))
        br = solve_best_response(
            inst, player=i, x_others=X,
            time_limit=br_time_limit, verbose=False,
        )
        pi_hat = float(br.profit)

        if pi_cur <= 0.0:
            if pi_hat > 0:
                return float("inf")
            ratios.append(1.0)
        else:
            ratios.append(pi_hat / pi_cur)

    return float(max(ratios)) if ratios else 1.0


# =====================================================================
# Random feasible profile generation
# =====================================================================

def random_feasible_profile(
    inst: GKGInstance,
    rng: random.Random,
) -> np.ndarray:
    """Generate a random feasible profile X (n x m).

    Strategy: process players in random order; each player solves their
    best response against the current partial allocation.  This guarantees
    both knapsack and shared-capacity constraints are satisfied.
    """
    n, m = inst.n, inst.m
    X = np.zeros((n, m), dtype=int)

    order = list(range(n))
    rng.shuffle(order)

    for i in order:
        br = solve_best_response(
            inst, player=i, x_others=X,
            time_limit=10.0, verbose=False,
        )
        X[i] = br.x_hat

    return X


# =====================================================================
# Random-Restart Best-Response Dynamics (RR-BRD)
# =====================================================================

def brd_random_restart(
    inst: GKGInstance,
    *,
    max_init: int = 5,
    max_round: int = 20,
    seed: int = 0,
    br_time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, bool, float]:
    """Random-restart BRD for GKG.

    Matches the NFG/DFG RR-BRD pattern:
    - Runs up to ``max_init`` restarts (random initial profiles).
    - Each restart runs up to ``max_round`` rounds with random player order.
    - If BRD converges (no player improves AND all BR solves succeeded),
      certify PNE and return immediately.
    - On the last restart, snapshot every end-of-round profile.
    - If no PNE is certified, evaluate alpha for every end-of-round profile
      from the last restart and return the one with minimum alpha.

    Returns ``(best_profile, pne_found, runtime)``.
    """
    t0 = time.time()
    rng = random.Random(seed)
    n, m = inst.n, inst.m

    best_profile = np.zeros((n, m), dtype=int)
    pne_found = False

    # End-of-round profiles from the last restart
    last_restart_end_profiles: List[np.ndarray] = []

    for init_idx in range(max_init):
        is_last_restart = (init_idx == max_init - 1)

        X = random_feasible_profile(inst, rng)

        for _round in range(max_round):
            changed = False
            all_solved = True

            order = list(range(n))
            rng.shuffle(order)

            for i in order:
                # Current profit
                pi_cur = sum(int(inst.p[i, j]) * int(X[i, j]) for j in range(m))

                # Best response
                br = solve_best_response(
                    inst, player=i, x_others=X,
                    time_limit=br_time_limit, verbose=False,
                )
                if br.profit is None:
                    all_solved = False
                    continue

                pi_hat = br.profit
                tol = 1e-9
                if pi_hat > pi_cur + tol:
                    X[i] = br.x_hat
                    changed = True

            # Snapshot end-of-round profile on last restart
            if is_last_restart:
                last_restart_end_profiles.append(X.copy())

            # Certify PNE only if nothing changed AND all BRs were solved
            if (not changed) and all_solved:
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
