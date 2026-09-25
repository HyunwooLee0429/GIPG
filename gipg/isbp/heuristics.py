from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import time
import random
import copy, math

import numpy as np

from .instance import ISBPInstance
from .best_response import BestResponseOracle
from .objectives import player_cost_from_profile


def _empty_profile(inst: ISBPInstance) -> Dict[int, Dict[int, int]]:
    return {i: {j: 0 for j in inst.bins} for i in inst.players}


def _normalize_profile(inst: ISBPInstance, x: Dict[int, Dict[int, Any]]) -> Dict[int, Dict[int, int]]:
    """Ensure x is a full (player->bin->int load) dict."""
    out = _empty_profile(inst)
    for i in inst.players:
        if i not in x:
            continue
        for j in inst.bins:
            if j in x[i]:
                out[i][j] = int(x[i][j])
    return out


def is_regular_instance(inst: ISBPInstance) -> bool:
    """Regular means all bins share the same (capacity, cost) and all players share the same weight."""
    caps = {int(inst.capacities[j]) for j in inst.bins}
    costs = {float(inst.costs[j]) for j in inst.bins}
    wts = {int(inst.weights[i]) for i in inst.players}
    return (len(caps) == 1) and (len(costs) == 1) and (len(wts) == 1)


def array_to_nested_dict(arr: np.ndarray) -> Dict[int, Dict[int, int]]:
    n, m = arr.shape
    return {i: {j: int(arr[i, j]) for j in range(m)} for i in range(n)}


# ---------------------------------------------------------------------------
# Random feasible initial profile (shared capacity)
# ---------------------------------------------------------------------------

def random_feasible_profile(inst: ISBPInstance, rng: random.Random, max_tries: int = 2000) -> Dict[int, Dict[int, int]]:
    """Generate a feasible profile that respects shared capacities."""
    players = list(inst.players)
    bins = list(inst.bins)

    for _ in range(max_tries):
        rem_cap = {j: int(inst.capacities[j]) for j in bins}
        x = _empty_profile(inst)

        ok = True
        for i in players:
            remaining = int(inst.weights[i])
            order = bins[:]
            rng.shuffle(order)

            for j in order:
                if remaining <= 0:
                    break
                if rem_cap[j] <= 0:
                    continue

                q = min(remaining, rem_cap[j])
                assign = rng.randint(1, q)  # ensure progress
                x[i][j] += assign
                rem_cap[j] -= assign
                remaining -= assign

            if remaining > 0:
                ok = False
                break

        if ok:
            return x

    return greedy_feasible_profile(inst)


def greedy_feasible_profile(inst: ISBPInstance) -> Dict[int, Dict[int, int]]:
    """Deterministic feasible profile: fill bins in order, one player at a time.

    Because items are splittable, a feasible profile exists whenever the total
    capacity is at least the total weight, and filling each bin to capacity
    before moving to the next one finds one. This is the fallback for the
    random construction above, which can fail on tight instances: it draws a
    random amount for each bin rather than the largest that fits, so on
    instances with little or no slack it may leave capacity stranded and never
    place the last player.
    """
    rem_cap = {j: int(inst.capacities[j]) for j in inst.bins}
    x = _empty_profile(inst)
    bins = list(inst.bins)
    k = 0

    for i in inst.players:
        remaining = int(inst.weights[i])
        while remaining > 0:
            while k < len(bins) and rem_cap[bins[k]] <= 0:
                k += 1
            if k >= len(bins):
                raise RuntimeError(
                    "infeasible instance: total capacity is below total weight"
                )
            j = bins[k]
            take = min(remaining, rem_cap[j])
            x[i][j] += take
            rem_cap[j] -= take
            remaining -= take
    return x


# ---------------------------------------------------------------------------
# Alpha computation for a given profile
# ---------------------------------------------------------------------------

def alpha_of_profile(
    inst: ISBPInstance,
    x_profile: Dict[int, Dict[int, int]],
    *,
    br_time_limit: Optional[float] = None,
) -> float:
    """Compute alpha(x) = max_i cost_i(x) / BR_i(x_-i)."""
    x_profile = _normalize_profile(inst, x_profile)
    totals = {j: sum(x_profile[i][j] for i in inst.players) for j in inst.bins}

    oracles = {i: BestResponseOracle(inst, i, log_to_console=0) for i in inst.players}

    worst = 1.0
    for i in inst.players:
        curr = player_cost_from_profile(
            players=inst.players,
            bins=inst.bins,
            costs=inst.costs,
            x_profile=x_profile,
            player=i,
        )
        opp = {j: int(totals[j] - x_profile[i][j]) for j in inst.bins}
        _br_alloc, br_obj = oracles[i].solve(opp, time_limit=br_time_limit)
        if br_obj is None or br_obj <= 0:
            return float("inf")
        worst = max(worst, float(curr) / float(br_obj))
    return float(worst)


# ---------------------------------------------------------------------------
# BRD heuristic (random-restart best-response dynamics)
# ---------------------------------------------------------------------------

def brd_random_restart(
    inst: ISBPInstance,
    *,
    max_init: int = 5,
    max_round: int = 20,
    seed: int = 0,
    br_time_limit: Optional[float] = None,
    end_round_profiles_out: Optional[List[Dict[int, Dict[int, int]]]] = None,
) -> Tuple[Dict[int, Dict[int, int]], bool, float]:
    """Random-restart BRD.

    - Runs up to `max_init` restarts.
    - Each restart runs up to `max_round` rounds.
      A round = one full pass over a random permutation of players.
    - If we *certify* a PNE (no change in a round AND all BR solves succeeded), return immediately.

    Requested behavior:
      - If no PNE is certified, then on the *last* restart only, we keep all end-of-round profiles
        and return the profile among them with the smallest alpha.
    """
    t0 = time.time()
    rng = random.Random(seed)

    # Build BR oracles once
    oracles = {i: BestResponseOracle(inst, i, log_to_console=0) for i in inst.players}

    def _deepcopy_profile(x: Dict[int, Dict[int, int]]) -> Dict[int, Dict[int, int]]:
        return {i: dict(x[i]) for i in inst.players}

    def _alpha_with_oracles(x_prof: Dict[int, Dict[int, int]]) -> float:
        """Compute alpha using the already-built oracles."""
        x_prof = _normalize_profile(inst, x_prof)
        totals_local = {j: sum(x_prof[i][j] for i in inst.players) for j in inst.bins}

        worst = 1.0
        for i in inst.players:
            curr = player_cost_from_profile(
                players=inst.players,
                bins=inst.bins,
                costs=inst.costs,
                x_profile=x_prof,
                player=i,
            )
            opp = {j: int(totals_local[j] - x_prof[i][j]) for j in inst.bins}
            _br_alloc, br_obj = oracles[i].solve(opp, time_limit=br_time_limit)
            if br_obj is None or br_obj <= 0:
                return float("inf")
            worst = max(worst, float(curr) / float(br_obj))
        return float(worst)

    # We only keep end-of-round profiles for the last restart
    last_restart_end_profiles: List[Dict[int, Dict[int, int]]] = []

    best_profile: Dict[int, Dict[int, int]] = _empty_profile(inst)
    best_pne = False

    for init_idx in range(max_init):
        is_last_restart = (init_idx == max_init - 1)

        x = _normalize_profile(inst, random_feasible_profile(inst, rng))

        for _round in range(max_round):
            changed = False
            all_solved = True  # needed to *certify* PNE

            order = list(inst.players)
            rng.shuffle(order)

            # precompute bin totals for speed
            totals = {j: sum(x[i][j] for i in inst.players) for j in inst.bins}

            for i in order:
                opp_load = {j: int(totals[j] - x[i][j]) for j in inst.bins}

                br_alloc, br_obj = oracles[i].solve(opp_load, time_limit=br_time_limit)
                if br_obj is None:
                    all_solved = False
                    continue

                # current cost under x (minimization)
                curr_cost = player_cost_from_profile(
                    players=inst.players,
                    bins=inst.bins,
                    costs=inst.costs,
                    x_profile=x,
                    player=i,
                )

                # update only if BR strictly improves
                tol = 1e-9
                if float(br_obj) < float(curr_cost) - tol:
                    br_alloc = _normalize_profile(inst, {i: br_alloc})[i]

                    # only now we should check equality / apply the update
                    if br_alloc != x[i]:
                        for j in inst.bins:
                            totals[j] += int(br_alloc[j]) - int(x[i][j])
                        x[i] = br_alloc
                        changed = True
                # else: keep x[i] (even if returned BR solution is different but not better)

            # end-of-round profile
            if is_last_restart:
                snap = _deepcopy_profile(x)
                last_restart_end_profiles.append(snap)
                if end_round_profiles_out is not None:
                    end_round_profiles_out.append(_deepcopy_profile(x))

            # certify PNE only if nothing changed AND all BRs were solved
            if (not changed) and all_solved:
                best_profile = x
                best_pne = True
                return best_profile, best_pne, time.time() - t0

        # no certified PNE in this restart; keep last profile as fallback
        best_profile = x

    # No certified PNE across restarts:
    # select best alpha among end-of-round profiles from *last* restart (if any)
    if last_restart_end_profiles:
        best_a = float("inf")
        best_x = last_restart_end_profiles[-1]
        for prof in last_restart_end_profiles:
            a = _alpha_with_oracles(prof)
            if a < best_a:
                best_a = a
                best_x = prof
        best_profile = best_x

    return _normalize_profile(inst, best_profile), best_pne, time.time() - t0


# ---------------------------------------------------------------------------
# Unified heuristic runner
# ---------------------------------------------------------------------------

def run_all_heuristics(
    inst: ISBPInstance,
    *,
    regular: Optional[bool] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the heuristics described in the computation note and pick the best alpha.

    Tie-breaking rule requested:
      - BRD wins ties in alpha.
      - Another heuristic wins only if it has strictly smaller alpha than BRD.

    Returns a dict:
      {
        "BRD": {..., "end_round_profiles_last_restart": [...]},
        "best": {name, x, alpha, runtime},
      }
    """
    if regular is None:
        regular = is_regular_instance(inst)

    results: Dict[str, Any] = {}

    # BRD as heuristic (works for both; may return certified PNE)
    brd_t0 = time.time()
    end_round_profiles: List[Dict[int, Dict[int, int]]] = []
    try:
        x_brd, pne_found, rt_brd_only = brd_random_restart(inst, seed=seed, end_round_profiles_out=end_round_profiles)
        # include alpha computation time in BRD runtime (requested)
        a_brd = alpha_of_profile(inst, x_brd)
        results["BRD"] = {
            "x": x_brd,
            "alpha": a_brd,
            "runtime": float(time.time() - brd_t0),
            "ok": True,
            "pne_found": bool(pne_found),
            "end_round_profiles_last_restart": end_round_profiles,
        }
    except Exception as e:
        results["BRD"] = {
            "x": None,
            "alpha": float("inf"),
            "runtime": float(time.time() - brd_t0),
            "ok": False,
            "error": repr(e),
            "pne_found": False,
            "end_round_profiles_last_restart": end_round_profiles,
        }

    feasible = {k: v for k, v in results.items() if v.get("ok", False)}
    if not feasible:
        results["best"] = {"name": None, "x": None, "alpha": float("inf"), "runtime": float("inf")}
        results["regular"] = bool(regular)
        return results

    # Requested selection logic:
    # If BRD is feasible, it wins unless some heuristic has strictly smaller alpha.
    if "BRD" in feasible:
        brd_alpha = float(feasible["BRD"]["alpha"])
        better_than_brd = {k: v for k, v in feasible.items() if (brd_alpha - float(v["alpha"]) >= 1e-4 ) }
        if better_than_brd:
            best_name = min(better_than_brd.keys(), key=lambda k: (float(better_than_brd[k]["alpha"]), float(better_than_brd[k]["runtime"])))
        else:
            best_name = "BRD"
    else:
        # fallback: min alpha then runtime
        best_name = min(feasible.keys(), key=lambda k: (float(feasible[k]["alpha"]), float(feasible[k]["runtime"])))

    best = feasible[best_name]
    results["best"] = {"name": best_name, "x": best["x"], "alpha": best["alpha"], "runtime": best["runtime"]}
    results["regular"] = bool(regular)
    return results


