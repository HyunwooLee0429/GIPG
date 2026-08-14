"""Heuristic algorithms for bilinear KPG: RR-BRD and alpha computation."""
from __future__ import annotations

from typing import List, Tuple, Optional
import time
import random

import numpy as np

from .instance import KPGInstance
from .best_response import solve_best_response
from .objectives import player_utility


# =====================================================================
# Alpha computation
# =====================================================================

def alpha_of_profile(
    inst: KPGInstance,
    X: np.ndarray,
    *,
    br_time_limit: Optional[float] = None,
) -> float:
    """Compute alpha(X) = max_i  BR_profit / current_utility.

    For maximisation: alpha = 1 means exact PNE, alpha > 1 means approximate.
    """
    n, m = inst.n, inst.m
    ratios: List[float] = []

    for i in range(n):
        pi_cur = player_utility(inst, X, i)
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
    inst: KPGInstance,
    rng: random.Random,
) -> np.ndarray:
    """Generate a random feasible profile X (n x m).

    Strategy: process players in random order; each solves BR against
    current partial allocation.  Guarantees knapsack + shared-capacity feasibility.
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
    inst: KPGInstance,
    *,
    max_init: int = 5,
    max_round: int = 20,
    seed: int = 0,
    br_time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, bool, float]:
    """Random-restart BRD for bilinear KPG.

    - Runs up to max_init restarts (random initial profiles).
    - Each restart runs up to max_round rounds with random player order.
    - If BRD converges (no player improves AND all BRs solved), certify PNE.
    - On last restart, snapshot every end-of-round profile.
    - If no PNE, evaluate alpha for snapshots, return best.

    Returns (best_profile, pne_found, runtime).
    """
    t0 = time.time()
    rng = random.Random(seed)
    n, m = inst.n, inst.m

    best_profile = np.zeros((n, m), dtype=int)
    pne_found = False

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
                pi_cur = player_utility(inst, X, i)

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

            if is_last_restart:
                last_restart_end_profiles.append(X.copy())

            if (not changed) and all_solved:
                return X.copy(), True, time.time() - t0

        best_profile = X.copy()

    # No certified PNE: select best alpha among last restart's snapshots
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
