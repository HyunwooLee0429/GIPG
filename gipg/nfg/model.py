"""Master MILP model for the Network Formation Game (used by GZR).

We build a *discretised* formulation analogous to the ISBP/DFG one-hot encoding:

    y[i,e,ell,s]  = 1  iff player i sends s units on edge e and total load is ell
    z[e,ell]      = 1  iff edge e carries total load ell
    x[i,e]        = integer flow of player i on edge e  (auxiliary, linked to y)

This encoding linearises the nonlinear Shapley cost:
    cost contribution = w_e * s / ell * y[i,e,ell,s]

CEI activation uses Activation B (per-state residual check via y-variables)
rather than z_over indicators, ensuring equilibrium preservation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Set

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

import numpy as np

from .instance import NFGInstance


# ------------------------------------------------------------------
# Fundamental cycle basis (undirected view of directed graph)
# ------------------------------------------------------------------

def _fundamental_cycle_basis(n_nodes: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Compute a fundamental cycle basis of the undirected view of the graph.

    Returns a list of cycles, where each cycle is a list of edge indices
    (referring to the *directed* edge list).  The number of fundamental
    cycles equals |E| - |V'| + 1  where V' is the number of nodes
    actually present in the graph.

    Algorithm:
      1. Build an undirected adjacency from the directed edge list.
      2. BFS spanning tree.
      3. For each non-tree edge, recover the unique tree-path cycle.
    """
    # Build undirected adjacency: node -> list of (neighbour, edge_index)
    adj: Dict[int, List[Tuple[int, int]]] = {v: [] for v in range(n_nodes)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    # Identify nodes actually in the graph
    graph_nodes: Set[int] = set()
    for u, v in edges:
        graph_nodes.add(u)
        graph_nodes.add(v)

    if not graph_nodes:
        return []

    # BFS spanning tree from smallest node
    root = min(graph_nodes)
    parent: Dict[int, int] = {root: -1}        # node -> parent node
    parent_edge: Dict[int, int] = {root: -1}   # node -> edge index to parent
    visited: Set[int] = {root}
    queue: deque = deque([root])
    tree_edges: Set[int] = set()

    while queue:
        u = queue.popleft()
        for v, eidx in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                parent_edge[v] = eidx
                tree_edges.add(eidx)
                queue.append(v)

    # For each non-tree edge, find the fundamental cycle
    cycles: List[List[int]] = []
    edge_set_undirected: Dict[Tuple[int, int], int] = {}
    for idx, (u, v) in enumerate(edges):
        key = (min(u, v), max(u, v))
        edge_set_undirected[key] = idx

    for eidx, (u, v) in enumerate(edges):
        if eidx in tree_edges:
            continue
        if u not in visited or v not in visited:
            continue  # skip isolated components

        # Trace path from u to root and v to root, find LCA
        path_u: List[int] = []
        ancestors_u: Dict[int, int] = {}  # node -> depth
        cur = u
        depth = 0
        while cur != -1:
            ancestors_u[cur] = depth
            path_u.append(cur)
            cur = parent.get(cur, -1)
            depth += 1

        # Trace from v upward to find LCA
        path_v: List[int] = []
        cur = v
        while cur not in ancestors_u:
            path_v.append(cur)
            cur = parent.get(cur, -1)
        lca = cur

        # Build cycle edge indices: u->...->lca<-...<-v, plus edge (u,v)
        cycle_edges: List[int] = [eidx]  # the non-tree edge itself

        # u up to lca
        cur = u
        while cur != lca:
            cycle_edges.append(parent_edge[cur])
            cur = parent[cur]

        # v up to lca
        cur = v
        while cur != lca:
            cycle_edges.append(parent_edge[cur])
            cur = parent[cur]

        cycles.append(cycle_edges)

    return cycles


@dataclass
class NFGModel:
    inst: NFGInstance
    model: "gp.Model"
    x: Dict[Tuple[int, int], "gp.Var"]               # (i, e) -> var
    y: Dict[Tuple[int, int, int, int], "gp.Var"]      # (i, e, ell, s) -> var
    z: Dict[Tuple[int, int], "gp.Var"]                 # (e, ell) -> var

    def extract_profile(self) -> np.ndarray:
        """Extract integer profile X from model solution."""
        n, m = self.inst.n_players, self.inst.n_edges
        X = np.zeros((n, m), dtype=int)
        for i in range(n):
            for e in range(m):
                X[i, e] = int(round(self.x[(i, e)].X))
        return X


def build_master_model(
    inst: NFGInstance,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
    add_vest_cut: bool = True,
) -> NFGModel:
    """Build the discretised master model for GZR.

    Objective: minimise social cost under Shapley sharing.
    Since social cost = sum of w_e for each used edge, we minimise:
        sum_{i,e,ell,s}  (w_e * s / ell) * y[i,e,ell,s]

    Parameters
    ----------
    add_vest_cut : bool
        If True, add weaker VEST cuts based on fundamental cycle basis.
        For each fundamental cycle C and player i, these cut off
        shared-unsaturated states:
          - G_i^C <= 1   (at most one shared-unsaturated edge per cycle)
          - g_{i,e1} + f_{i,e2} <= 1  for e1 != e2 in C
        where g = shared-unsaturated (s>=1, s<ell, ell<c_e),
              h = solo-unsaturated (s=ell>=1, ell<c_e), f = g + h.
    """
    if gp is None:
        raise ImportError("gurobipy is required")

    n = inst.n_players
    E = inst.n_edges

    mdl = gp.Model("NFG_master")
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.LazyConstraints = 1
    mdl.Params.Threads = 16 if threads is None else int(threads)
    if time_limit is not None:
        mdl.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        mdl.Params.MIPGap = float(mip_gap)

    # ------------------------------------------------------------------
    # Pre-compute index ranges
    # ------------------------------------------------------------------
    # L[e] = list of feasible total loads {0,...,c_e}
    # S_max[(i,e)] = max flow of player i on edge e = min(d_i, c_e)
    L = {e: list(range(int(inst.capacities[e]) + 1)) for e in range(E)}
    S_max = {
        (i, e): min(int(inst.demands[i]), int(inst.capacities[e]))
        for i in range(n) for e in range(E)
    }

    # Feasible (s) values for player i on edge e given total load ell
    def s_range(i: int, e: int, ell: int) -> List[int]:
        return list(range(min(ell, S_max[(i, e)]) + 1))

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    x: Dict[Tuple[int, int], gp.Var] = {}
    y: Dict[Tuple[int, int, int, int], gp.Var] = {}
    z: Dict[Tuple[int, int], gp.Var] = {}

    for i in range(n):
        for e in range(E):
            x[(i, e)] = mdl.addVar(
                vtype=GRB.INTEGER, lb=0, ub=S_max[(i, e)],
                name=f"x[{i},{e}]",
            )

    for e in range(E):
        for ell in L[e]:
            z[(e, ell)] = mdl.addVar(vtype=GRB.BINARY, name=f"z[{e},{ell}]")

    for i in range(n):
        for e in range(E):
            for ell in L[e]:
                for s in s_range(i, e, ell):
                    y[(i, e, ell, s)] = mdl.addVar(
                        vtype=GRB.BINARY, name=f"y[{i},{e},{ell},{s}]",
                    )

    mdl.update()

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # (C1) Choose one (ell, s) per (i, e)
    for i in range(n):
        for e in range(E):
            mdl.addConstr(
                gp.quicksum(
                    y[(i, e, ell, s)]
                    for ell in L[e]
                    for s in s_range(i, e, ell)
                ) == 1,
                name=f"choose[{i},{e}]",
            )

    # (C2) Link x <-> y:  x[i,e] = sum_{ell,s} s * y[i,e,ell,s]
    for i in range(n):
        for e in range(E):
            mdl.addConstr(
                gp.quicksum(
                    s * y[(i, e, ell, s)]
                    for ell in L[e]
                    for s in s_range(i, e, ell)
                ) == x[(i, e)],
                name=f"link[{i},{e}]",
            )

    # (C3) Flow conservation per player: A x_i = b_i
    for i in range(n):
        b_i = inst.demand_vector(i)
        for v in range(inst.n_nodes):
            mdl.addConstr(
                gp.quicksum(x[(i, e)] for e in inst.outgoing(v))
                - gp.quicksum(x[(i, e)] for e in inst.incoming(v))
                == int(b_i[v]),
                name=f"flow[{i},{v}]",
            )

    # (C4) y implies z:  y[i,e,ell,s] <= z[e,ell]
    for e in range(E):
        for ell in L[e]:
            for i in range(n):
                for s in s_range(i, e, ell):
                    mdl.addConstr(
                        y[(i, e, ell, s)] <= z[(e, ell)],
                        name=f"y_z[{i},{e},{ell},{s}]",
                    )

    # (C5) Load consistency:  sum_i sum_s s*y[i,e,ell,s] = ell * z[e,ell]
    for e in range(E):
        for ell in L[e]:
            mdl.addConstr(
                gp.quicksum(
                    s * y[(i, e, ell, s)]
                    for i in range(n)
                    for s in s_range(i, e, ell)
                ) == ell * z[(e, ell)],
                name=f"load[{e},{ell}]",
            )

    # (C6) Exactly one load level per edge
    for e in range(E):
        mdl.addConstr(
            gp.quicksum(z[(e, ell)] for ell in L[e]) == 1,
            name=f"oneload[{e}]",
        )

    # ------------------------------------------------------------------
    # Weaker VEST cuts (fundamental cycle basis) — Lemma 2 / Eq. (22)
    # ------------------------------------------------------------------
    # Lemma 2: If player i uses an undirected cycle C where ALL arcs
    # are unsaturated, then no other player uses any arc in C.
    # Equivalently: if all arcs in C are unsaturated for player i
    # (F_i^C = |C|), then none can be shared-unsaturated (G_i^C = 0).
    #
    # Big-M-free formulation (Eq. 22):
    #   g_{i,e} + (G_i^C + H_i^C - |C|) <= 0     for all e in C
    #   i.e.  g_{i,e} + F_i^C <= |C|
    #
    # where:
    #   g_{i,e} = shared-unsaturated: s >= 1, s < ell, ell < c_e
    #   h_{i,e} = solo-unsaturated:   s = ell >= 1, ell < c_e
    #   f_{i,e} = g_{i,e} + h_{i,e}  (unsaturated with positive flow)
    #   F_i^C  = sum_{e in C} f_{i,e}
    if add_vest_cut:
        cycles = _fundamental_cycle_basis(inst.n_nodes, list(inst.edges))
        n_vest_cuts = 0
        for c_idx, cycle_edges in enumerate(cycles):
            cycle_len = len(cycle_edges)
            for i in range(n):
                g_terms: Dict[int, List] = {}  # edge -> list of y-vars (shared-unsat)
                f_terms: Dict[int, List] = {}  # edge -> list of y-vars (any unsat)
                for e in cycle_edges:
                    c_e = int(inst.capacities[e])
                    gj: List = []
                    hj: List = []
                    for ell in L[e]:
                        if ell < 1 or ell >= c_e:
                            continue  # need ell < c_e and ell >= 1
                        for s in s_range(i, e, ell):
                            if s >= 1 and s < ell:
                                # shared-unsaturated: s >= 1, s < ell, ell < c_e
                                gj.append(y[(i, e, ell, s)])
                            elif s == ell and s >= 1:
                                # solo-unsaturated: s = ell >= 1, ell < c_e
                                hj.append(y[(i, e, ell, s)])
                    g_terms[e] = gj
                    f_terms[e] = gj + hj  # f = g + h

                # F_i^C = sum_{e in C} f_{i,e}
                all_f = [v for e in cycle_edges for v in f_terms[e]]

                # Eq. (22): g_{i,e} + F_i^C <= |C|  for each e in C
                for e in cycle_edges:
                    if not g_terms[e]:
                        continue
                    if not all_f:
                        continue
                    mdl.addConstr(
                        gp.quicksum(g_terms[e]) + gp.quicksum(all_f) <= cycle_len,
                        name=f"wvest[c{c_idx},i{i},e{e}]",
                    )
                    n_vest_cuts += 1

        if verbose:
            print(f"[NFG model] Added {n_vest_cuts} weaker VEST cuts "
                  f"from {len(cycles)} fundamental cycles")

    # ------------------------------------------------------------------
    # Objective: minimise social cost under Shapley sharing
    # ------------------------------------------------------------------
    # cost = sum_{i,e,ell>0,s>0} (w_e * s / ell) * y[i,e,ell,s]
    obj = gp.LinExpr()
    for i in range(n):
        for e in range(E):
            w_e = float(inst.edge_costs[e])
            for ell in L[e]:
                if ell == 0:
                    continue
                for s in s_range(i, e, ell):
                    if s == 0:
                        continue
                    coef = w_e * float(s) / float(ell)
                    obj.addTerms(coef, y[(i, e, ell, s)])
    mdl.setObjective(obj, GRB.MINIMIZE)

    return NFGModel(inst=inst, model=mdl, x=x, y=y, z=z)
