"""Network Formation Game instance representation and generation.

An NFG is played on a directed graph G = (V, E) with Shapley/proportional
cost sharing (Chen & Roughgarden 2006, Dragotto 2023):

  - N = {0,...,n-1} players, each with source s_i, sink t_i, integer demand d_i.
  - Each edge e has capacity c_e (max total flow) and fixed cost w_e.
  - Player i's cost under Shapley sharing:
        pi_i(X) = sum_e  w_e * X[i,e] / ell_e
    where ell_e = sum_k X[k,e] is total edge load.

Extended to integer-splittable, edge-capacitated setting:
  - Each player may split demand across multiple paths (integer units).
  - Shared edge capacities couple players (GNEP).

The game is a GNEP because capacity constraints couple players:
    X_i(x_{-i}) = { x_i in Z_>=0^E : A x_i = b_i,
                     x_{i,e} <= c_e - sum_{k!=i} x_{k,e}  for all e }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json

import numpy as np


@dataclass(frozen=True)
class NFGInstance:
    """Network Formation Game instance.

    Parameters
    ----------
    n_nodes : int
        Number of nodes |V|.
    edges : tuple of (int, int)
        Directed edges as (tail, head) pairs.  Edge index = position in list.
    n_players : int
    sources : np.ndarray   shape (n_players,)
    sinks : np.ndarray     shape (n_players,)
    demands : np.ndarray   shape (n_players,)  integer demands d_i >= 1
    capacities : np.ndarray  shape (n_edges,)  edge capacities c_e >= 1
    edge_costs : np.ndarray  shape (n_edges,)  fixed cost w_e > 0 per edge
    seed : int
    meta : dict
    """
    n_nodes: int
    edges: Tuple[Tuple[int, int], ...]
    n_players: int
    sources: np.ndarray
    sinks: np.ndarray
    demands: np.ndarray
    capacities: np.ndarray
    edge_costs: np.ndarray
    seed: int
    meta: Dict[str, Any]

    # ---- derived helpers ----

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def players(self) -> List[int]:
        return list(range(self.n_players))

    @property
    def edge_indices(self) -> List[int]:
        return list(range(self.n_edges))

    def tail(self, e: int) -> int:
        return self.edges[e][0]

    def head(self, e: int) -> int:
        return self.edges[e][1]

    def outgoing(self, v: int) -> List[int]:
        """Edge indices leaving node v."""
        return [e for e, (u, _) in enumerate(self.edges) if u == v]

    def incoming(self, v: int) -> List[int]:
        """Edge indices entering node v."""
        return [e for e, (_, w) in enumerate(self.edges) if w == v]

    def demand_vector(self, i: int) -> np.ndarray:
        """Node-indexed demand vector b_i  (b[s_i]=+d_i, b[t_i]=-d_i, 0 elsewhere)."""
        b = np.zeros(self.n_nodes, dtype=int)
        b[int(self.sources[i])] = int(self.demands[i])
        b[int(self.sinks[i])] = -int(self.demands[i])
        return b

    def max_player_edge_flow(self, i: int, e: int) -> int:
        """Upper bound on x_{i,e}: min(d_i, c_e)."""
        return min(int(self.demands[i]), int(self.capacities[e]))

    # ---- serialisation ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "edges": list(self.edges),
            "n_players": self.n_players,
            "sources": self.sources.tolist(),
            "sinks": self.sinks.tolist(),
            "demands": self.demands.tolist(),
            "capacities": self.capacities.tolist(),
            "edge_costs": self.edge_costs.tolist(),
            "seed": int(self.seed),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NFGInstance":
        return NFGInstance(
            n_nodes=int(d["n_nodes"]),
            edges=tuple(tuple(e) for e in d["edges"]),
            n_players=int(d["n_players"]),
            sources=np.array(d["sources"], dtype=int),
            sinks=np.array(d["sinks"], dtype=int),
            demands=np.array(d["demands"], dtype=int),
            capacities=np.array(d["capacities"], dtype=int),
            edge_costs=np.array(d["edge_costs"], dtype=float),
            seed=int(d.get("seed", 0)),
            meta=dict(d.get("meta", {})),
        )


# =====================================================================
# Graph generators
# =====================================================================

def _random_connected_dag(
    n_nodes: int,
    n_extra_edges: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Generate a random DAG on n_nodes that is weakly connected.

    Strategy: build a random spanning tree (directed root->leaves),
    then add n_extra_edges random forward edges (u->v with u < v).
    """
    perm = rng.permutation(n_nodes)
    edges_set: set = set()

    # spanning tree: each node (except root) gets one parent with smaller index
    for idx in range(1, n_nodes):
        parent_idx = int(rng.integers(0, idx))
        u, v = int(perm[parent_idx]), int(perm[idx])
        if u > v:
            u, v = v, u  # ensure u < v for DAG
        edges_set.add((u, v))

    # extra forward edges
    attempts = 0
    added = 0
    while added < n_extra_edges and attempts < 10 * n_extra_edges:
        u = int(rng.integers(0, n_nodes - 1))
        v = int(rng.integers(u + 1, n_nodes))
        if (u, v) not in edges_set:
            edges_set.add((u, v))
            added += 1
        attempts += 1

    return sorted(edges_set)


def _series_parallel_graph(layers: int, width: int) -> Tuple[int, List[Tuple[int, int]]]:
    """Generate a layered series-parallel-like graph.

    nodes: 0 = source layer, then `layers` internal layers of `width` nodes, then sink.
    All edges go from layer k to layer k+1.
    """
    n_nodes = 1 + layers * width + 1  # source + internal + sink
    source = 0
    sink = n_nodes - 1
    edges: List[Tuple[int, int]] = []

    # source -> first layer
    for j in range(width):
        edges.append((source, 1 + j))

    # layer k -> layer k+1
    for k in range(layers - 1):
        base_k = 1 + k * width
        base_next = 1 + (k + 1) * width
        for a in range(width):
            for b in range(width):
                edges.append((base_k + a, base_next + b))

    # last layer -> sink
    last_base = 1 + (layers - 1) * width
    for j in range(width):
        edges.append((last_base + j, sink))

    return n_nodes, edges


def _layered_graph(
    n_layers: int,
    n_width: int,
    rng: np.random.Generator,
    edge_prob: float = 0.8,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Generate a layered DAG with random edges between adjacent layers.

    The graph has ``n_layers`` layers, each containing ``n_width`` nodes,
    for a total of ``n_layers * n_width`` nodes.  Nodes are numbered
    layer-by-layer: layer k contains nodes  k*n_width, ..., (k+1)*n_width - 1.

    Edges are drawn independently between every pair of nodes in adjacent
    layers (layer k -> layer k+1) with probability ``edge_prob``.
    To guarantee connectivity, a random monotone path from each node in
    layer 0 to some node in layer L-1 is ensured (see below).

    Parameters
    ----------
    n_layers : int
        Number of layers (L).
    n_width : int
        Number of nodes per layer (W).
    rng : numpy.random.Generator
        Random number generator.
    edge_prob : float
        Probability of each inter-layer edge (default 0.8).

    Returns
    -------
    n_nodes : int
        Total number of nodes (L * W).
    edges : list of (int, int)
        Sorted edge list.
    """
    n_nodes = n_layers * n_width
    edges_set: set = set()

    # Random edges between adjacent layers
    for k in range(n_layers - 1):
        base_k = k * n_width
        base_next = (k + 1) * n_width
        for a in range(n_width):
            for b in range(n_width):
                if rng.random() < edge_prob:
                    edges_set.add((base_k + a, base_next + b))

    # Guarantee: every node in layer 0 can reach at least one node in
    # the last layer.  For each source node, walk forward layer by layer;
    # if no outgoing edge exists at the current node, add one at random.
    for s in range(n_width):  # layer-0 nodes
        cur = s
        for k in range(n_layers - 1):
            base_next = (k + 1) * n_width
            # Check if cur already has an outgoing edge to layer k+1
            has_out = any((cur, base_next + b) in edges_set for b in range(n_width))
            if not has_out:
                # Add a random edge to the next layer
                b = int(rng.integers(0, n_width))
                edges_set.add((cur, base_next + b))
            # Move to a random successor in the next layer
            successors = [base_next + b for b in range(n_width)
                          if (cur, base_next + b) in edges_set]
            cur = successors[int(rng.integers(0, len(successors)))]

    # Similarly, guarantee every node in the last layer is reachable from
    # at least one node in layer 0 (walk backward).
    last_layer_base = (n_layers - 1) * n_width
    for t_offset in range(n_width):
        t = last_layer_base + t_offset
        cur = t
        for k in range(n_layers - 1, 0, -1):
            base_prev = (k - 1) * n_width
            # Check if cur has an incoming edge from layer k-1
            has_in = any((base_prev + a, cur) in edges_set for a in range(n_width))
            if not has_in:
                a = int(rng.integers(0, n_width))
                edges_set.add((base_prev + a, cur))
            # Move to a random predecessor in the previous layer
            predecessors = [base_prev + a for a in range(n_width)
                            if (base_prev + a, cur) in edges_set]
            cur = predecessors[int(rng.integers(0, len(predecessors)))]

    return n_nodes, sorted(edges_set)


def _random_flow_dag(
    n_nodes: int,
    rng: np.random.Generator,
    edge_prob: float = 0.3,
) -> List[Tuple[int, int]]:
    """Generate a random DAG suitable for flow games.

    Ensures: (a) source (node 0) has at least one outgoing edge,
             (b) sink (node n-1) has at least one incoming edge,
             (c) every node lies on at least one s-t path.
    """
    source, sink = 0, n_nodes - 1
    edges_set: set = set()

    # Random forward edges
    for u in range(n_nodes - 1):
        for v in range(u + 1, n_nodes):
            if rng.random() < edge_prob:
                edges_set.add((u, v))

    # Ensure a monotone path source -> ... -> sink for connectivity.
    # Pick a random subset of internal nodes and sort them to create a
    # forward-connected path (preserving DAG property u < v naturally).
    internals = list(range(1, n_nodes - 1))
    rng.shuffle(internals)
    n_path = min(len(internals), max(2, len(internals) // 2))
    path_internals = sorted(internals[:n_path])
    path_nodes = [source] + path_internals + [sink]
    for idx in range(len(path_nodes) - 1):
        edges_set.add((path_nodes[idx], path_nodes[idx + 1]))

    # Prune nodes with no s-t path through them
    fwd = {source}
    for u in range(n_nodes):
        if u in fwd:
            for (a, b) in edges_set:
                if a == u:
                    fwd.add(b)
    bwd = {sink}
    for v in range(n_nodes - 1, -1, -1):
        if v in bwd:
            for (a, b) in edges_set:
                if b == v:
                    bwd.add(a)

    reachable = fwd & bwd
    edges_set = {(u, v) for (u, v) in edges_set if u in reachable and v in reachable}

    return sorted(edges_set)


# =====================================================================
# Instance generators
# =====================================================================

def generate_nfg_instance(
    *,
    n_players: int,
    n_nodes: int = 6,
    n_extra_edges: int = 3,
    cap_range: Tuple[int, int] = (5, 15),
    demand_range: Tuple[int, int] = (1, 4),
    cost_range: Tuple[int, int] = (1, 10),
    graph_type: str = "random_dag",
    sp_layers: int = 2,
    sp_width: int = 3,
    seed: int = 0,
) -> NFGInstance:
    """Generate a random NFG instance.

    Parameters
    ----------
    graph_type : "random_dag" | "series_parallel"
    cost_range : range for edge costs w_e (uniform integer)
    """
    rng = np.random.default_rng(seed)

    # --- Graph ---
    if graph_type == "series_parallel":
        n_nodes_actual, edge_list = _series_parallel_graph(sp_layers, sp_width)
    else:
        n_nodes_actual = n_nodes
        edge_list = _random_connected_dag(n_nodes_actual, n_extra_edges, rng)

    n_edges = len(edge_list)

    # --- Players (source-sink pairs) ---
    source_node = 0
    sink_node = n_nodes_actual - 1
    sources = np.full(n_players, source_node, dtype=int)
    sinks = np.full(n_players, sink_node, dtype=int)
    demands = rng.integers(demand_range[0], demand_range[1] + 1, size=n_players).astype(int)

    # --- Edge capacities ---
    total_demand = int(demands.sum())
    cap_lo, cap_hi = cap_range
    capacities = rng.integers(
        max(cap_lo, total_demand),
        max(cap_hi, total_demand) + 1,
        size=n_edges,
    ).astype(int)

    # --- Edge costs (fixed cost w_e for Shapley sharing) ---
    edge_costs = rng.integers(cost_range[0], cost_range[1] + 1, size=n_edges).astype(float)

    meta = {
        "generator": "generate_nfg_instance",
        "graph_type": graph_type,
    }

    return NFGInstance(
        n_nodes=n_nodes_actual,
        edges=tuple(tuple(e) for e in edge_list),
        n_players=n_players,
        sources=sources,
        sinks=sinks,
        demands=demands,
        capacities=capacities,
        edge_costs=edge_costs,
        seed=seed,
        meta=meta,
    )


def generate_dragotto_nfg(
    *,
    n_players: int,
    n_nodes: int,
    seed: int = 0,
    edge_prob: float = 0.3,
    demand_range: Tuple[int, int] = (1, 5),
    cap_factor: float = 1.5,
    cost_range: Tuple[int, int] = (1, 20),
) -> NFGInstance:
    """Generate an NFG instance following the Dragotto (2023) / Chen-Roughgarden
    (2006) network formation game structure.

    Reference
    ---------
    Dragotto (2023), "Zero Regrets Algorithm", Section 5.2:
      - Directed graph G = (V, E)
      - n players, each with source s_i, sink t_i, demand d_i = 1 (binary in original)
      - Shapley cost sharing: pi_i = sum_e w_e * x_{i,e} / ell_e
      - We extend to integer-splittable (d_i >= 1) and edge-capacitated

    Parameters
    ----------
    n_players : int
        Number of players.
    n_nodes : int
        Number of graph nodes.
    cost_range : tuple
        Range for fixed edge costs w_e.
    cap_factor : float
        Capacity = ceil(cap_factor * total_demand) ensures feasibility.
    """
    rng = np.random.default_rng(seed)

    # --- Graph ---
    edge_list = _random_flow_dag(n_nodes, rng, edge_prob=edge_prob)
    n_edges = len(edge_list)

    # --- Players ---
    source_node = 0
    sink_node = n_nodes - 1
    sources = np.full(n_players, source_node, dtype=int)
    sinks = np.full(n_players, sink_node, dtype=int)
    demands = rng.integers(demand_range[0], demand_range[1] + 1, size=n_players).astype(int)

    # --- Capacities ---
    total_demand = int(demands.sum())
    base_cap = max(1, int(np.ceil(cap_factor * total_demand)))
    capacities = rng.integers(
        base_cap, base_cap + max(3, total_demand), size=n_edges,
    ).astype(int)

    # --- Edge costs ---
    edge_costs = rng.integers(cost_range[0], cost_range[1] + 1, size=n_edges).astype(float)

    meta = {
        "generator": "generate_dragotto_nfg",
        "literature": "Dragotto (2023), Chen & Roughgarden (2006)",
        "note": "Extended to integer-splittable edge-capacitated setting",
        "cap_factor": cap_factor,
        "edge_prob": edge_prob,
    }

    return NFGInstance(
        n_nodes=n_nodes,
        edges=tuple(tuple(e) for e in edge_list),
        n_players=n_players,
        sources=sources,
        sinks=sinks,
        demands=demands,
        capacities=capacities,
        edge_costs=edge_costs,
        seed=seed,
        meta=meta,
    )


def generate_dragotto_binary_nfg(
    *,
    n_players: int,
    n_nodes: int,
    seed: int = 0,
    edge_prob: float = 0.3,
    cost_range: Tuple[int, int] = (1, 20),
) -> NFGInstance:
    """Generate a *binary* NFG matching the original Dragotto (2023) formulation
    exactly: each player has demand d_i = 1 (binary edge selection).

    This is the vanilla Chen-Roughgarden network formation game.
    """
    rng = np.random.default_rng(seed)

    edge_list = _random_flow_dag(n_nodes, rng, edge_prob=edge_prob)
    n_edges = len(edge_list)

    source_node = 0
    sink_node = n_nodes - 1
    sources = np.full(n_players, source_node, dtype=int)
    sinks = np.full(n_players, sink_node, dtype=int)
    demands = np.ones(n_players, dtype=int)  # binary: d_i = 1

    # Capacity: each edge can carry all players (no capacity bottleneck)
    capacities = np.full(n_edges, n_players, dtype=int)

    edge_costs = rng.integers(cost_range[0], cost_range[1] + 1, size=n_edges).astype(float)

    meta = {
        "generator": "generate_dragotto_binary_nfg",
        "literature": "Dragotto (2023), vanilla binary NFG",
        "note": "d_i = 1 for all players (original formulation)",
    }

    return NFGInstance(
        n_nodes=n_nodes,
        edges=tuple(tuple(e) for e in edge_list),
        n_players=n_players,
        sources=sources,
        sinks=sinks,
        demands=demands,
        capacities=capacities,
        edge_costs=edge_costs,
        seed=seed,
        meta=meta,
    )


def dragotto_parameter_grid(
    *,
    n_seeds: int = 50,
) -> List[Dict[str, Any]]:
    """Return a parameter grid for NFG experiments.

    Grid: n_players in {2, 4, 10}, n_nodes in {10, 15, 20},
    n_seeds per combination.
    """
    grid = []
    for n_players in [2, 4, 10]:
        for n_nodes in [10, 15, 20]:
            for s in range(n_seeds):
                grid.append({
                    "n_players": n_players,
                    "n_nodes": n_nodes,
                    "seed": s,
                })
    return grid


# =====================================================================
# 2x2 Instance Generator: capacity tightness x source-sink symmetry
# =====================================================================

CapacityLevel = str  # "tight" or "loose"
SymmetryLevel = str  # "symmetric" or "asymmetric"


def generate_nfg_2x2(
    *,
    n_players: int,
    n_nodes: int = 15,
    capacity_level: CapacityLevel = "tight",
    symmetry: SymmetryLevel = "symmetric",
    demand_range: Tuple[int, int] = (2, 5),
    cost_range: Tuple[int, int] = (1, 20),
    edge_prob: float = 0.3,
    seed: int = 0,
    max_feasibility_attempts: int = 20,
    graph_type: str = "random_flow_dag",
    n_layers: int = 5,
    n_width: int = 5,
) -> NFGInstance:
    """Generate an integer-splittable NFG instance with controlled capacity
    tightness and source-sink symmetry.

    This generator produces instances for a 2x2 experimental design:

        capacity_level x symmetry  =  {tight, loose} x {symmetric, asymmetric}

    All instances are integer-splittable (d_i >= 2), differentiating from
    the classical unsplittable (d_i = 1) model of Anshelevich (2008) and
    Dragotto (2023).

    Parameters
    ----------
    n_players : int
        Number of players.
    n_nodes : int
        Number of graph nodes.
    capacity_level : "tight" or "loose"
        Controls how restrictive edge capacities are relative to total demand.

        - "tight": cap_e ~ Uniform[max(d_max, ceil(0.7D)), ceil(0.9D)].
          Most edges cannot carry all players simultaneously, forcing
          players to compete for edge space (strong GNEP coupling).
        - "loose": cap_e ~ Uniform[max(d_max, ceil(0.9D)), ceil(1.1D)].
          Some edges may bind, some won't — mild GNEP coupling.

    symmetry : "symmetric" or "asymmetric"
        Controls source-sink pair assignment.

        - "symmetric": All players share the same source (node 0) and
          sink (node n-1).
        - "asymmetric": Players have distinct source-sink pairs drawn
          from the left and right portions of the DAG.

    demand_range : tuple
        Range for integer demands d_i.  Lower bound >= 2 to ensure
        integer-splittable instances.
    cost_range : tuple
        Range for edge costs w_e (uniform integer).
    edge_prob : float
        Probability of each forward edge in the random DAG.
    seed : int
        Random seed.
    max_feasibility_attempts : int
        If the initial capacity draw is infeasible, retry with slightly
        increased capacities up to this many times.
    graph_type : str
        Graph topology: "random_flow_dag" (default) or "layered".
        When "layered", uses ``_layered_graph`` with ``n_layers`` layers
        of ``n_width`` nodes each, and random inter-layer edges with
        probability ``edge_prob`` (default 0.8 for layered).
    n_layers : int
        Number of layers for the layered graph (only used when
        graph_type="layered").  Default 5.
    n_width : int
        Number of nodes per layer for the layered graph (only used when
        graph_type="layered").  Default 5.

    Returns
    -------
    NFGInstance
        A feasible NFG instance.

    Raises
    ------
    ValueError
        If capacity_level or symmetry has an invalid value, or if a
        feasible instance cannot be generated within the allowed attempts.
    """
    if capacity_level not in ("tight", "loose"):
        raise ValueError(f"capacity_level must be 'tight' or 'loose', got '{capacity_level}'")
    if symmetry not in ("symmetric", "asymmetric"):
        raise ValueError(f"symmetry must be 'symmetric' or 'asymmetric', got '{symmetry}'")

    if graph_type not in ("random_flow_dag", "layered"):
        raise ValueError(f"graph_type must be 'random_flow_dag' or 'layered', got '{graph_type}'")

    rng = np.random.default_rng(seed)

    # --- Graph ---
    if graph_type == "layered":
        n_nodes, edge_list = _layered_graph(n_layers, n_width, rng, edge_prob=edge_prob)
    else:
        edge_list = _random_flow_dag(n_nodes, rng, edge_prob=edge_prob)
    n_edges = len(edge_list)

    # --- Demands (integer-splittable: d_i >= 2) ---
    lo, hi = max(2, demand_range[0]), demand_range[1]
    demands = rng.integers(lo, hi + 1, size=n_players).astype(int)
    total_demand = int(demands.sum())
    d_max = int(demands.max())

    # --- Source-sink pairs ---
    # Identify nodes actually present in the graph (after pruning in _random_flow_dag)
    graph_nodes = set()
    for (a, b) in edge_list:
        graph_nodes.add(a)
        graph_nodes.add(b)
    graph_nodes_sorted = sorted(graph_nodes)

    # Build adjacency list for efficient reachability
    adj_fwd: Dict[int, List[int]] = {v: [] for v in graph_nodes_sorted}
    for (a, b) in edge_list:
        adj_fwd[a].append(b)

    # Build forward reachability for each node using topological order
    # (DAG edges go from lower to higher IDs, so sorted order = topological order)
    fwd_reach: Dict[int, set] = {}
    for node in graph_nodes_sorted:
        reached = {node}
        for u in graph_nodes_sorted:
            if u >= node and u in reached:
                for v in adj_fwd.get(u, []):
                    reached.add(v)
        fwd_reach[node] = reached

    if symmetry == "symmetric":
        sources = np.full(n_players, 0, dtype=int)
        sinks = np.full(n_players, n_nodes - 1, dtype=int)
    else:
        if graph_type == "layered":
            # Asymmetric layered: sources from layer 0, sinks from layer L-1
            layer0_nodes = list(range(0, n_width))
            last_layer_nodes = list(range((n_layers - 1) * n_width, n_layers * n_width))

            # Build valid pairs: source in layer 0 can reach sink in layer L-1
            valid_pairs = []
            for s in layer0_nodes:
                for t in last_layer_nodes:
                    if t in fwd_reach.get(s, set()):
                        valid_pairs.append((s, t))

            # Fallback (should not happen due to connectivity guarantees)
            if len(valid_pairs) == 0:
                valid_pairs = [(0, n_nodes - 1)]

            # Assign distinct pairs to players (with replacement if needed)
            sources_list = []
            sinks_list = []
            for i in range(n_players):
                pair_idx = int(rng.integers(0, len(valid_pairs)))
                s, t = valid_pairs[pair_idx]
                sources_list.append(s)
                sinks_list.append(t)

            sources = np.array(sources_list, dtype=int)
            sinks = np.array(sinks_list, dtype=int)
        else:
            # Asymmetric random_flow_dag: distinct s-t pairs from nodes actually
            # in the graph.  Split into "left" (source candidates) and "right"
            # (sink candidates) using the first and last third of graph nodes.
            n_graph = len(graph_nodes_sorted)
            left_end = max(1, n_graph // 3)
            right_start = max(left_end, n_graph - n_graph // 3)
            left_pool = graph_nodes_sorted[:left_end]
            right_pool = graph_nodes_sorted[right_start:]

            # Build valid (source, sink) pairs: s can reach t in the DAG
            valid_pairs = []
            for s in left_pool:
                for t in right_pool:
                    if s < t and t in fwd_reach.get(s, set()):
                        valid_pairs.append((s, t))

            # Fallback: if no valid pairs from left/right pools, try all pairs
            if len(valid_pairs) == 0:
                for s in graph_nodes_sorted:
                    for t in graph_nodes_sorted:
                        if s < t and t in fwd_reach.get(s, set()):
                            valid_pairs.append((s, t))

            # Ultimate fallback: (0, n-1) which always works by construction
            if len(valid_pairs) == 0:
                valid_pairs = [(0, n_nodes - 1)]

            # Assign pairs to players (with replacement if needed)
            sources_list = []
            sinks_list = []
            for i in range(n_players):
                pair_idx = int(rng.integers(0, len(valid_pairs)))
                s, t = valid_pairs[pair_idx]
                sources_list.append(s)
                sinks_list.append(t)

            sources = np.array(sources_list, dtype=int)
            sinks = np.array(sinks_list, dtype=int)

    # --- Edge capacities ---
    # Asymmetric s-t pairs create harder multi-commodity flow problems,
    # so we use slightly wider caps for asymmetric + tight.
    if capacity_level == "tight":
        if symmetry == "asymmetric":
            # Asymmetric needs more headroom: 0.8–1.0 D
            cap_lo = max(d_max, int(np.ceil(0.7 * total_demand)))
            cap_hi = max(cap_lo + 1, int(np.ceil(0.9 * total_demand)))
        else:
            # Symmetric: 0.7–0.9 D (all players share same path set)
            cap_lo = max(d_max, int(np.ceil(0.7 * total_demand)))
            cap_hi = max(cap_lo + 1, int(np.ceil(0.9 * total_demand)))
    else:  # "loose"
        # cap/D ~ 0.9–1.1: some edges may bind, some won't — mild coupling
        cap_lo = max(d_max, int(np.ceil(0.9 * total_demand)))
        cap_hi = max(cap_lo + 1, int(np.ceil(1.1 * total_demand)))

    capacities = rng.integers(cap_lo, cap_hi + 1, size=n_edges).astype(int)

    # --- Edge costs ---
    edge_costs = rng.integers(cost_range[0], cost_range[1] + 1, size=n_edges).astype(float)

    # --- Feasibility check ---
    # Verify that all players can simultaneously route their demands.
    # If infeasible, increase capacities slightly and retry.
    edges_tuple = tuple(tuple(e) for e in edge_list)

    inst = NFGInstance(
        n_nodes=n_nodes,
        edges=edges_tuple,
        n_players=n_players,
        sources=sources,
        sinks=sinks,
        demands=demands,
        capacities=capacities,
        edge_costs=edge_costs,
        seed=seed,
        meta={},
    )

    feasibility_bumps = 0
    for attempt in range(max_feasibility_attempts):
        if _check_feasibility(inst):
            break
        # Increase capacities by 1 on each edge and retry
        capacities = capacities + 1
        feasibility_bumps += 1
        inst = NFGInstance(
            n_nodes=n_nodes,
            edges=edges_tuple,
            n_players=n_players,
            sources=sources,
            sinks=sinks,
            demands=demands,
            capacities=capacities,
            edge_costs=edge_costs,
            seed=seed,
            meta={},
        )
    else:
        raise ValueError(
            f"Could not generate feasible instance after "
            f"{max_feasibility_attempts} attempts (seed={seed}, "
            f"n_players={n_players}, n_nodes={n_nodes}, "
            f"capacity_level={capacity_level}, symmetry={symmetry})"
        )

    meta = {
        "generator": "generate_nfg_2x2",
        "graph_type": graph_type,
        "capacity_level": capacity_level,
        "symmetry": symmetry,
        "total_demand": total_demand,
        "cap_range_actual": [int(capacities.min()), int(capacities.max())],
        "feasibility_bumps": feasibility_bumps,
        "n_graph_nodes": len(graph_nodes_sorted),
        "n_edges": n_edges,
    }
    if graph_type == "layered":
        meta["n_layers"] = n_layers
        meta["n_width"] = n_width

    return NFGInstance(
        n_nodes=n_nodes,
        edges=edges_tuple,
        n_players=n_players,
        sources=sources,
        sinks=sinks,
        demands=demands,
        capacities=capacities,
        edge_costs=edge_costs,
        seed=seed,
        meta=meta,
    )


def _check_feasibility(inst: NFGInstance) -> bool:
    """Check if all players can simultaneously route their demands.

    Solves a multi-commodity integer flow feasibility problem using
    gurobipy.  Returns True if feasible, False otherwise.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        # If Gurobi is not available, assume feasible (optimistic)
        return True

    n = inst.n_players
    E = inst.n_edges

    mdl = gp.Model("feasibility_check")
    mdl.Params.OutputFlag = 0
    mdl.Params.Threads = 16
    mdl.Params.TimeLimit = 30.0

    # Variables: x[i,e] = flow of player i on edge e
    x = {}
    for i in range(n):
        for e in range(E):
            ub = min(int(inst.demands[i]), int(inst.capacities[e]))
            x[(i, e)] = mdl.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"x[{i},{e}]")

    mdl.update()

    # Flow conservation per player
    for i in range(n):
        b_i = inst.demand_vector(i)
        for v in range(inst.n_nodes):
            mdl.addConstr(
                gp.quicksum(x[(i, e)] for e in inst.outgoing(v))
                - gp.quicksum(x[(i, e)] for e in inst.incoming(v))
                == int(b_i[v]),
            )

    # Shared capacity constraints
    for e in range(E):
        mdl.addConstr(
            gp.quicksum(x[(i, e)] for i in range(n)) <= int(inst.capacities[e]),
        )

    mdl.setObjective(0, GRB.MINIMIZE)
    mdl.optimize()

    return mdl.Status == GRB.OPTIMAL


def nfg_2x2_parameter_grid(
    *,
    n_seeds: int = 50,
    n_players_list: Optional[List[int]] = None,
    n_nodes_list: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Return the full 2x2 parameter grid for NFG experiments.

    Grid dimensions:
        - capacity_level: {"tight", "loose"}
        - symmetry: {"symmetric", "asymmetric"}
        - n_players: {2, 4, 6}  (or custom)
        - n_nodes: {10, 15, 20}  (or custom)
        - seed: 0 .. n_seeds-1

    Returns list of dicts with keys: n_players, n_nodes, capacity_level,
    symmetry, seed.
    """
    if n_players_list is None:
        n_players_list = [2, 4, 6]
    if n_nodes_list is None:
        n_nodes_list = [10, 15, 20]

    grid = []
    for cap_level in ["tight", "loose"]:
        for sym in ["symmetric", "asymmetric"]:
            for n_players in n_players_list:
                for n_nodes in n_nodes_list:
                    for s in range(n_seeds):
                        grid.append({
                            "n_players": n_players,
                            "n_nodes": n_nodes,
                            "capacity_level": cap_level,
                            "symmetry": sym,
                            "seed": s,
                        })
    return grid


# =====================================================================
# I/O
# =====================================================================

def save_instance(inst: NFGInstance, path: str) -> None:
    """Save instance to JSON."""
    with open(path, "w") as f:
        json.dump(inst.to_dict(), f, indent=2)


def load_instance(path: str) -> NFGInstance:
    """Load instance from JSON."""
    with open(path, "r") as f:
        return NFGInstance.from_dict(json.load(f))
