from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set, Any
import time, math

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

from .instance import ISBPInstance, identical_bin_groups
from .heuristics import is_regular_instance
from .best_response import BestResponseOracle
from .cuts import (
    build_equilibrium_inequality_expr,
    symmetric_br_patterns,
    deviation_cost_from_opponents_load,
)


# ---------------------------------------------------------------------------
# Base model builder
# ---------------------------------------------------------------------------

def build_full_discretized_model(
        inst: ISBPInstance,
        *,
        add_bin_symmetry_breaking: bool = True,   # order load across identical bins
        add_vest_cut: bool = True,                 # VEST cut (valid inequality, not symmetry-breaking)
        add_player_anchoring: bool = False,        # anchor players to bins (regular instances, w <= u)
        threads: Optional[int] = None,
        log_to_console: int = 0,
) -> Tuple[
    "gp.Model",
    Dict[Tuple[int, int, int], "gp.Var"],
    Dict[Tuple[int, int, int, int], "gp.Var"],
    Dict[Tuple[int, int], "gp.Var"],
    List[List[int]],
]:
    """
    Build the fully discretized ISBP model (x,y,z).

    This is the base feasibility/assignment model. Equilibrium enforcement is done via
    equilibrium inequalities added as lazy constraints in a callback.

    Returns
    -------
    (model, x, y, z, groups)
        - model: Gurobi model
        - x[i,j,s]: player i assigns s units to bin j
        - y[i,j,t,s]: player i assigns s units to bin j when total load is t
        - z[j,t]: bin j activated at total load t
        - groups: list of identical-bin groups (by (u_j, c_j)), used by symmetry-breaking cuts.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    players, bins = inst.players, inst.bins
    costs, capacities, weights, T, S = inst.costs, inst.capacities, inst.weights, inst.T, inst.S

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("ZR_ISBP", env=env)
    model.Params.Threads = 16 if threads is None else int(threads)
    if log_to_console:
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = log_to_console

    # Decision variables
    x: Dict[Tuple[int, int, int], gp.Var] = {}
    y: Dict[Tuple[int, int, int, int], gp.Var] = {}
    z: Dict[Tuple[int, int], gp.Var] = {}

    # x[i,j,s]
    for i in players:
        for j in bins:
            s_max = min(int(capacities[j]), int(weights[i]))
            for s in range(s_max + 1):
                x[(i, j, s)] = model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j},{s}]")

    # y[i,j,t,s]
    for i in players:
        for j in bins:
            for t in T[j]:
                for s in S[i][j][t]:
                    y[(i, j, t, s)] = model.addVar(vtype=GRB.BINARY, name=f"y[{i},{j},{t},{s}]")

    # z[j,t]
    for j in bins:
        for t in T[j]:
            z[(j, t)] = model.addVar(vtype=GRB.BINARY, name=f"z[{j},{t}]")

    # Objective: total cost
    model.setObjective(
        gp.quicksum(
            costs[j] * (s / t) * y[(i, j, t, s)]
            for i in players
            for j in bins
            for t in T[j]
            for s in S[i][j][t]
            if t != 0
        ),
        GRB.MINIMIZE,
    )

    # (1) For each (i,j), choose exactly one (t,s)
    for i in players:
        for j in bins:
            model.addConstr(
                gp.quicksum(y[(i, j, t, s)] for t in T[j] for s in S[i][j][t]) == 1,
                name=f"choose_one[{i},{j}]",
            )

            # (2) Link x and y: x[i,j,s] = sum_{t>=s} y[i,j,t,s]
            s_max = min(int(capacities[j]), int(weights[i]))
            for s in range(s_max + 1):
                model.addConstr(
                    gp.quicksum(y[(i, j, t2, s)] for t2 in T[j] if t2 >= s) == x[(i, j, s)],
                    name=f"link_xy[{i},{j},{s}]",
                )

    # (3) Each player meets their weight
    for i in players:
        model.addConstr(
            gp.quicksum(s * y[(i, j, t, s)] for j in bins for t in T[j] for s in S[i][j][t])
            == weights[i],
            name=f"weight[{i}]",
        )

    # (4) Occupancy coupling + bin activation
    for j in bins:
        for t in T[j]:
            # y implies z
            for i in players:
                model.addConstr(
                    gp.quicksum(y[(i, j, t, s)] for s in S[i][j][t]) <= z[(j, t)],
                    name=f"y_implies_z[{i},{j},{t}]",
                )
            # load consistency
            model.addConstr(
                gp.quicksum(s * y[(i, j, t, s)] for i in players for s in S[i][j][t])
                == t * z[(j, t)],
                name=f"load_consistency[{j},{t}]",
            )
        # at most one occupancy per bin
        model.addConstr(
            gp.quicksum(z[(j, t)] for t in T[j]) == 1,
            name=f"bin_single_occupancy[{j}]",
        )

    # Groups of identical bins (by (u_j, c_j))
    groups = identical_bin_groups(inst)

    # ------------------------------------------------------------
    # Break bin-label symmetry: order total load within each identical-bin group.
    # Within each group of bins sharing the same (u_j, c_j), order by
    # non-increasing total load.
    # ------------------------------------------------------------
    if add_bin_symmetry_breaking and any(len(D) > 1 for D in groups):
        for gi, D in enumerate(groups):
            if len(D) <= 1:
                continue
            for ell in range(len(D) - 1):
                a, b = D[ell], D[ell + 1]
                L_a = gp.quicksum(int(t) * z[(a, t)] for t in T[a])
                L_b = gp.quicksum(int(t) * z[(b, t)] for t in T[b])
                model.addConstr(L_a >= L_b, name=f"bin_load[g{gi},{ell}]")

    # ------------------------------------------------------------
    # VEST cuts (valid inequalities, not symmetry-breaking)
    #
    # Regular instances (identical costs, capacities, weights):
    #   VEST cut (Thm 6.3): F_i <= 1
    #   At most one unsaturated bin per player.
    #
    # Irregular instances:
    #   Weaker VEST cut: cuts off shared-unsaturated states.
    #   A bin j is "shared-unsaturated" for player i when the bin is
    #   not full (t < u_j) and player i shares it with others (s < t).
    #   A bin j is "solo-unsaturated" when t < u_j and s = t (player
    #   alone).  At a PNE, each player uses at most one shared-
    #   unsaturated bin; if such a bin exists, every other unsaturated
    #   bin must be solo.
    #
    #   Indicator sums per player i and bin j:
    #     g_{i,j} = shared-unsaturated: s >= 1, s < t, t < u_j
    #     h_{i,j} = solo-unsaturated:   s = t >= 1, t < u_j
    #     f_{i,j} = g_{i,j} + h_{i,j}   (unsaturated, either type)
    #
    #   Cuts (big-M-free form):
    #     G_i <= 1                          (at most one shared-unsaturated bin)
    #     g_{i,j1} + f_{i,j2} <= 1  ∀j1≠j2 (shared-unsat excludes other unsat)
    # ------------------------------------------------------------
    if add_vest_cut:
        regular = is_regular_instance(inst)
        const_cost = len({float(costs[j]) for j in bins}) == 1

        if regular and const_cost:
            # --- VEST cut: F_i <= 1 (regular-uniform only) ---
            for i in players:
                model.addConstr(
                    gp.quicksum(
                        y[(i, j, t, s)]
                        for j in bins
                        for t in T[j]
                        if 1 <= int(t) < int(capacities[j])  # unsaturated: t<u_j and t>0
                        for s in S[i][j][t]
                        if int(s) > 0
                    )
                    <= 1,
                    name=f"vest[{i}]",
                )
        else:
            # --- Weaker VEST cut: cut off shared-unsaturated states ---
            for i in players:
                # Build per-bin indicator expressions
                g_terms = {}  # g_{i,j}: shared-unsaturated
                f_terms = {}  # f_{i,j}: unsaturated (shared or solo)

                for j in bins:
                    u_j = int(capacities[j])

                    # g_{i,j}: bin j shared-unsaturated for player i
                    #   s >= 1, s < t, t < u_j  (player shares with others)
                    gj = [
                        y[(i, j, t, s)]
                        for t in T[j]
                        if 2 <= int(t) < u_j          # t >= 2 needed for s >= 1 and s < t
                        for s in S[i][j][t]
                        if 1 <= int(s) < int(t)
                    ]
                    # h_{i,j}: bin j solo-unsaturated for player i
                    #   s = t >= 1, t < u_j  (player occupies bin alone)
                    hj = [
                        y[(i, j, t, s)]
                        for t in T[j]
                        if 1 <= int(t) < u_j
                        for s in S[i][j][t]
                        if int(s) == int(t) and int(s) >= 1
                    ]
                    g_terms[j] = gj
                    f_terms[j] = gj + hj  # f = g + h

                # G_i <= 1: at most one shared-unsaturated bin per player
                all_g = [v for j in bins for v in g_terms[j]]
                if all_g:
                    model.addConstr(
                        gp.quicksum(all_g) <= 1,
                        name=f"weak_vest_G[{i}]",
                    )

                # g_{i,j1} + f_{i,j2} <= 1: if j1 is shared-unsaturated,
                # no other bin j2 can be unsaturated
                for j1 in bins:
                    if not g_terms[j1]:
                        continue
                    for j2 in bins:
                        if j2 == j1 or not f_terms[j2]:
                            continue
                        model.addConstr(
                            gp.quicksum(g_terms[j1]) + gp.quicksum(f_terms[j2]) <= 1,
                            name=f"weak_vest_gf[{i},{j1},{j2}]",
                        )

    # ------------------------------------------------------------
    # Break player-label symmetry: anchor designated players to designated bins.
    # For regular instances with w <= u, player i must put >0 in bin i
    # for i = 0..k-1 where k = min(n, ceil(n*w/u)).
    # Only valid when w <= u (each player fits in a single bin).
    # ------------------------------------------------------------
    if add_player_anchoring:
        regular_bins = (len({int(capacities[j]) for j in bins}) == 1) and (len({float(costs[j]) for j in bins}) == 1)
        regular_players = (len({int(weights[i]) for i in players}) == 1)
        w_leq_u = regular_bins and regular_players and (int(weights[0]) <= int(capacities[0]))
        if regular_bins and regular_players and w_leq_u:
            smallest_m = math.ceil(len(players) * weights[0] / capacities[0])
            k = min(len(players), smallest_m)
            for i in range(k):
                s_max = min(int(weights[i]), int(capacities[i]))
                model.addConstr(gp.quicksum(x[(i, i, s)] for s in range(1, s_max + 1)) >= 1,
                                name=f"player_anchor[{i},{i}]")

    model.update()

    return model, x, y, z, groups


# ---------------------------------------------------------------------------
# Utilities for extracting/valuations
# ---------------------------------------------------------------------------

def _extract_profile_from_x_solution(
        inst: ISBPInstance,
        x_vals: Dict[Tuple[int, int, int], float],
) -> Dict[int, Dict[int, int]]:
    """Extract (player -> bin -> load) from incumbent values of x."""
    x_prof = {i: {j: 0 for j in inst.bins} for i in inst.players}
    for i in inst.players:
        for j in inst.bins:
            s_max = min(int(inst.capacities[j]), int(inst.weights[i]))
            chosen = 0
            bestv = -1.0
            for s in range(s_max + 1):
                v = x_vals.get((i, j, s), 0.0)
                if v > bestv:
                    bestv = v
                    chosen = s
            x_prof[i][j] = int(chosen)
    return x_prof


def _player_cost_from_profile(
        inst: ISBPInstance,
        player: int,
        x_prof: Dict[int, Dict[int, int]],
) -> float:
    """Compute player cost from x-profile using cost sharing c_j * s / t."""
    total_load = {j: sum(x_prof[i][j] for i in inst.players) for j in inst.bins}
    cost = 0.0
    for j in inst.bins:
        t = total_load[j]
        s = x_prof[player][j]
        if t > 0 and s > 0:
            cost += float(inst.costs[j]) * (float(s) / float(t))
    return float(cost)


def _br_signature(player: int, br: Dict[int, int], tag: str = "EI") -> Tuple[Any, ...]:
    """Canonical signature for a best-response pattern (for deduplication)."""
    items = tuple(sorted((j, int(s)) for j, s in br.items() if int(s) > 0))
    return (tag, int(player), items)


# ---------------------------------------------------------------------------
# Zero-regret with lazy constraints + BR reuse
# ---------------------------------------------------------------------------

def solve_gzr(
        inst: ISBPInstance,
        *,
        alpha: float = 1.0,
        bigM: Optional[float] = None,
        time_limit: float = 1800.0,
        gurobi_params: Optional[Dict[str, Any]] = None,
        add_bin_symmetry_breaking: bool = True,  # order load across identical bins
        add_symmetric_eis: bool = True,          # CEIs generated from permuted best responses
        add_vest_cut: bool = True,               # VEST cut (valid inequality, not symmetry-breaking)
        add_player_anchoring: Optional[bool] = None,  # None = auto-select on regular instances
        auto_cut_selection: bool = True,

        sym_sample_fraction: float = 0.50,
        sym_max_combinations: int = 50,
        sym_max_permutations: int = 50,
        sym_cap_per_core: int = 10,
        violation_tol: float = 1e-9,
        stop_at_first_pne: bool = True,
        threads: Optional[int] = None,
        log_to_console: int = 1,
        br_cache: Optional[List[Dict[str, Any]]] = None,
        warm_start: Optional[Dict[int, Dict[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Zero-regret solve using *one* Gurobi optimize() with lazy constraints.

    Core guarantee:
      - In each MIPSOL callback, we compute (exact) best-responses for all players.
      - If no player has a violated α-PNE condition, the incumbent is an α-PNE.

    If `stop_at_first_pne=True`, we **terminate Gurobi immediately** once such an
    incumbent is found, and return that profile (even though Gurobi status will
    typically be GRB.INTERRUPTED).

    BR reuse across different alpha values is supported via `br_cache`:
        br_cache entries are dicts:
          {"player": i, "br_alloc": {...}, "alpha_min": alpha_min}
        We reuse entries with alpha_min <= alpha (safe by monotonicity).

    Returns a dict with detailed EI counts:
        - eis_added_core: EIs from true BR violations (separation)
        - eis_added_symmetric: symmetric-breaking EIs (optional)
        - eis_added_reused: EIs injected from br_cache before optimize()
        - eis_added_total: sum of the above
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    if bigM is None:
        bigM = float(sum(cost for _, cost in inst.costs.items()))

    if br_cache is None:
        br_cache = []

    # Auto-select cut configuration based on instance regularity.
    #
    # Regular (w<=u):
    #   bin-load ordering, symmetry-generated CEIs, and player anchoring
    #   + VEST cut (at most one unsaturated bin per player)
    #   When warm_start is provided (BRD+GZR), the warm-start profile is
    #   post-processed to satisfy (i) and (iii) before injection.
    #
    # Irregular:
    #   bin-load ordering and symmetry-generated CEIs
    #   + weaker VEST cut (cuts off shared-unsaturated states)
    # Player anchoring is only meaningful on regular instances, so it is
    # selected automatically unless the caller asked for it explicitly. Passing
    # add_player_anchoring overrides this choice; the w <= u guard is enforced
    # inside build_full_discretized_model either way.
    if auto_cut_selection and add_player_anchoring is None:
        add_player_anchoring = is_regular_instance(inst)
    elif add_player_anchoring is None:
        add_player_anchoring = False

    model, x, y, z, groups = build_full_discretized_model(
        inst,
        add_bin_symmetry_breaking=add_bin_symmetry_breaking,
        add_vest_cut=add_vest_cut,
        add_player_anchoring=add_player_anchoring,
        threads=threads,
        log_to_console=log_to_console,
    )

    # Required for cbLazy
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = float(time_limit)
    if gurobi_params:
        for k, v in gurobi_params.items():
            setattr(model.Params, k, v)

    # --- MIP warm-start from a profile ---
    if warm_start is not None:
        # Make a mutable copy so we don't modify the caller's dict
        warm_start = {i: dict(warm_start[i]) for i in inst.players}

        # Compute total loads per bin from the warm-start profile
        ws_total = {j: sum(warm_start[i][j] for i in inst.players) for j in inst.bins}

        # --- Permute warm-start within identical-bin groups to satisfy
        #     bin-load ordering: non-increasing total load.
        if add_bin_symmetry_breaking and any(len(D) > 1 for D in groups):
            for D in groups:
                if len(D) <= 1:
                    continue
                # source_bins[k] = which original bin should map to D[k]
                source_bins = sorted(D, key=lambda j: -ws_total[j])
                if source_bins == list(D):
                    continue  # already in correct order
                # Permute allocations
                old = {j: {i: warm_start[i][j] for i in inst.players} for j in D}
                for k, target_j in enumerate(D):
                    src_j = source_bins[k]
                    for i in inst.players:
                        warm_start[i][target_j] = old[src_j][i]
                # Update ws_total for permuted bins
                for j in D:
                    ws_total[j] = sum(warm_start[i][j] for i in inst.players)

        # --- Permute players to satisfy the player-anchoring constraint.
        #     For regular instances, all players are identical so permutation
        #     preserves the PNE.  Anchoring requires: player i has >0 in bin i
        #     for i = 0..k-1.
        if add_player_anchoring:
            _reg_bins = (len({int(inst.capacities[j]) for j in inst.bins}) == 1) and \
                        (len({float(inst.costs[j]) for j in inst.bins}) == 1)
            _reg_players = (len({int(inst.weights[i]) for i in inst.players}) == 1)
            if _reg_bins and _reg_players:
                k = min(len(inst.players),
                        math.ceil(len(inst.players) * int(inst.weights[0]) / int(inst.capacities[0])))
                # Bipartite matching: anchor position i -> player with >0 in bin i
                candidates = {i: [p for p in inst.players if warm_start[p][i] > 0]
                              for i in range(k)}
                # Greedy matching, most-constrained-first
                used: set = set()
                sigma: Dict[int, int] = {}  # sigma[position] = old_player
                for i in sorted(range(k), key=lambda i: len(candidates[i])):
                    for p in candidates[i]:
                        if p not in used:
                            sigma[i] = p
                            used.add(p)
                            break

                if len(sigma) == k:
                    # Build full permutation: perm[new_player] = old_player
                    remaining = [p for p in inst.players if p not in used]
                    perm = []
                    rem_idx = 0
                    for i in range(len(inst.players)):
                        if i in sigma:
                            perm.append(sigma[i])
                        else:
                            perm.append(remaining[rem_idx])
                            rem_idx += 1
                    # Apply permutation
                    old_ws = {i: dict(warm_start[i]) for i in inst.players}
                    for new_i, old_i in enumerate(perm):
                        warm_start[new_i] = dict(old_ws[old_i])

        for i in inst.players:
            for j in inst.bins:
                s_i = warm_start[i][j]
                t_j = ws_total[j]
                s_max = min(int(inst.capacities[j]), int(inst.weights[i]))
                # x[i,j,s]
                for s in range(s_max + 1):
                    if (i, j, s) in x:
                        x[(i, j, s)].Start = 1.0 if s == s_i else 0.0
                # y[i,j,t,s] and z[j,t]
                for t in inst.T[j]:
                    for s in inst.S[i][j][t]:
                        if (i, j, t, s) in y:
                            y[(i, j, t, s)].Start = 1.0 if (t == t_j and s == s_i) else 0.0
        for j in inst.bins:
            t_j = ws_total[j]
            for t in inst.T[j]:
                if (j, t) in z:
                    z[(j, t)].Start = 1.0 if t == t_j else 0.0

    # Per-player BR oracle (reused inside callback)
    br_oracle = {i: BestResponseOracle(inst, i, log_to_console=0) for i in inst.players}

    # Diagnostics + de-duplication
    eis_added_total = 0
    eis_added_core = 0
    eis_added_symmetric = 0
    eis_added_reused = 0

    added_signatures: Set[Tuple[Any, ...]] = set()
    first_pne_time: List[Optional[float]] = [None]  # mutable container for callback

    # ----------------------------------------------------------------------
    # 1. Reuse BRs from previous alpha values (alpha_min <= current alpha)
    # ----------------------------------------------------------------------
    for entry in br_cache:
        player = entry["player"]
        br_alloc = entry["br_alloc"]
        alpha_min = entry.get("alpha_min", 1.0)
        if alpha_min > alpha + 1e-9:
            continue

        sig = _br_signature(player, br_alloc, tag="EI_REUSE")
        if sig in added_signatures:
            continue

        lhs, rhs = build_equilibrium_inequality_expr(
            player=player,
            BR=br_alloc,
            alpha=alpha,
            bigM=bigM,
            y=y,
            costs=inst.costs,
            capacities=inst.capacities,
            T=inst.T,
            S=inst.S,
            bins=inst.bins,
        )
        model.addConstr(lhs <= rhs, name=f"EI_reuse_p{player}")
        added_signatures.add(sig)
        eis_added_reused += 1
        eis_added_total += 1

    start_time = time.time()

    # ----------------------------------------------------------------------
    # 2. Lazy callback: separate violated EIs on incumbents
    # ----------------------------------------------------------------------
    def callback(cb_model, where):  # pragma: no cover
        nonlocal br_cache, eis_added_total, eis_added_core, eis_added_symmetric

        if where != GRB.Callback.MIPSOL:
            return

        # Read incumbent x-solution and derive profile
        x_vals = {k: cb_model.cbGetSolution(var) for k, var in x.items()}
        x_prof = _extract_profile_from_x_solution(inst, x_vals)

        # Bin totals and per-player opponents loads
        total_load = {j: sum(x_prof[i][j] for i in inst.players) for j in inst.bins}

        any_violation = False

        for i in inst.players:
            curr_cost = _player_cost_from_profile(inst, i, x_prof)
            opp_load = {j: int(total_load[j] - x_prof[i][j]) for j in inst.bins}

            # Solve BR for player i against current opponents' loads
            br_alloc, br_obj = br_oracle[i].solve(opp_load)
            if br_obj is None:
                continue

            # Check violation of alpha-PNE condition for player i
            if curr_cost <= alpha * br_obj + violation_tol:
                continue

            any_violation = True

            # ---- Core EI for this BR ----
            sig = _br_signature(i, br_alloc, tag="EI")
            if sig not in added_signatures:
                lhs, rhs = build_equilibrium_inequality_expr(
                    player=i,
                    BR=br_alloc,
                    alpha=alpha,
                    bigM=bigM,
                    y=y,
                    costs=inst.costs,
                    capacities=inst.capacities,
                    T=inst.T,
                    S=inst.S,
                    bins=inst.bins,
                )
                cb_model.cbLazy(lhs <= rhs)
                added_signatures.add(sig)
                eis_added_core += 1
                eis_added_total += 1
                br_cache.append({"player": i, "br_alloc": br_alloc, "alpha_min": alpha})

            # ---- Optional: symmetric-breaking EIs ----
            if add_symmetric_eis:
                sym_added_this_core = 0
                SYM_CAP_PER_CORE = sym_cap_per_core  # <= 3 symmetric EIs per one core EI

                for br_sym in symmetric_br_patterns(
                        inst=inst,
                        br_alloc=br_alloc,
                        groups=groups,
                        sample_fraction=sym_sample_fraction,
                        max_combinations=sym_max_combinations,
                        max_permutations=sym_max_permutations,
                        rng_seed=0,
                ):
                    if sym_added_this_core >= SYM_CAP_PER_CORE:
                        break

                    sig2 = _br_signature(i, br_sym, tag="EI_SYM")
                    if sig2 in added_signatures:
                        continue

                    lhs2, rhs2 = build_equilibrium_inequality_expr(
                        player=i,
                        BR=br_sym,
                        alpha=alpha,
                        bigM=bigM,
                        y=y,
                        costs=inst.costs,
                        capacities=inst.capacities,
                        T=inst.T,
                        S=inst.S,
                        bins=inst.bins,
                    )
                    cb_model.cbLazy(lhs2 <= rhs2)
                    added_signatures.add(sig2)

                    sym_added_this_core += 1
                    eis_added_symmetric += 1
                    eis_added_total += 1

        # If no violations for any player, we have an α-PNE.
        if not any_violation:
            # Record time of first verified PNE (only once)
            if first_pne_time[0] is None:
                first_pne_time[0] = time.time() - start_time
            if stop_at_first_pne:
                # Store the incumbent profile so we can return it after termination.
                cb_model._pne_profile = x_prof
                cb_model._pne_alpha = float(alpha)
                cb_model.terminate()

    model.optimize(callback)

    runtime = time.time() - start_time
    gurobi_status = int(model.Status)

    # ----------------------------------------------------------------------
    # 3. Extract best available profile (callback-verified or incumbent)
    # ----------------------------------------------------------------------
    x_profile: Optional[Dict[int, Dict[int, int]]] = None

    if hasattr(model, "_pne_profile"):
        x_profile = getattr(model, "_pne_profile")
    elif model.SolCount > 0:
        x_prof = {i: {j: 0 for j in inst.bins} for i in inst.players}
        for i in inst.players:
            for j in inst.bins:
                s_max = min(int(inst.capacities[j]), int(inst.weights[i]))
                for s in range(s_max + 1):
                    if x[(i, j, s)].X > 0.5:
                        x_prof[i][j] = int(s)
                        break
        x_profile = x_prof

    # Map Gurobi status directly
    _STATUS_MAP = {
        GRB.OPTIMAL:     "OPTIMAL",
        GRB.INFEASIBLE:  "INFEASIBLE",
        GRB.TIME_LIMIT:  "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    status_str = _STATUS_MAP.get(gurobi_status, f"GUROBI_{gurobi_status}")

    # When stop_at_first_pne=True, the callback calls model.terminate()
    # after finding a verified PNE, so Gurobi reports INTERRUPTED.
    # Override to "FEASIBLE" for consistency with GKG/NFG.
    if stop_at_first_pne and hasattr(model, "_pne_profile"):
        status_str = "FEASIBLE"

    # MIPGap (requires at least one feasible solution).
    mip_gap = float(model.MIPGap) if model.SolCount > 0 else None
    # ObjBound — always attempt (useful even when SolCount == 0).
    try:
        obj_bound = float(model.ObjBound)
    except Exception:
        obj_bound = None

    return {
        "status": status_str,
        "gurobi_status": gurobi_status,
        "alpha": float(alpha),
        "runtime": float(runtime),
        "mip_gap": mip_gap,
        "obj_bound": obj_bound,
        "sol_count": int(model.SolCount),
        # Back-compat key:
        "cuts_added": int(eis_added_total),
        # Detailed EI counts:
        "eis_added_total": int(eis_added_total),
        "eis_added_core": int(eis_added_core),
        "eis_added_symmetric": int(eis_added_symmetric),
        "eis_added_reused": int(eis_added_reused),
        "x_profile": x_profile,
        "first_pne_time": first_pne_time[0],
        "br_cache": br_cache,
    }
