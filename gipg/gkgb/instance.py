"""Bilinear KPG instance: load Dragotto-format files + generate shared constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from itertools import permutations
import numpy as np
import pathlib


@dataclass(frozen=True)
class KPGInstance:
    """Bilinear Knapsack Problem Game instance with shared item capacities.

    n players, m items.

    Each player i has:
      - profits p[i,j] >= 0   (direct linear profit)
      - weights w[i,j] >= 0
      - knapsack capacity b[i] >= 0

    Bilinear coupling (interaction terms):
      - C[i,k,j]:  coefficient for x_j^i * x_j^k  in player i's objective,
        for all ordered pairs (i,k), i != k.

    Global coupling (shared constraints):
      - item capacities c[j] in {1,...,n}

    All arrays are integer-valued numpy arrays.
    """
    n: int
    m: int
    p: np.ndarray       # shape (n, m) — direct profits
    w: np.ndarray       # shape (n, m) — weights
    b: np.ndarray       # shape (n,)   — knapsack capacities
    C: np.ndarray       # shape (n, n, m) — interaction coefficients; C[i,i,:] = 0
    c: np.ndarray       # shape (m,)   — shared item capacities
    interaction_type: str   # "B" or "C"
    tag: str                # instance identifier (e.g. "2-100-5-cij")
    seed: int               # seed used for shared-constraint generation
    meta: Dict[str, Any]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def players(self) -> List[int]:
        return list(range(self.n))

    @property
    def items(self) -> List[int]:
        return list(range(self.m))

    @property
    def ordered_pairs(self) -> List[Tuple[int, int]]:
        return list(permutations(range(self.n), 2))

    def player_utility(self, X: np.ndarray, player: int) -> float:
        """Compute u_i(X) = direct profits + interaction terms."""
        i = player
        n, m = self.n, self.m
        util = float(np.dot(self.p[i], X[i]))
        for k in range(n):
            if k == i:
                continue
            # interaction: sum_j C[i,k,j] * x_j^i * x_j^k
            util += float(np.dot(self.C[i, k] * X[i], X[k]))
        return util

    def social_welfare(self, X: np.ndarray) -> float:
        """Total social welfare = sum_i u_i(X)."""
        return sum(self.player_utility(X, i) for i in range(self.n))


# ======================================================================
# Loading Dragotto-format instance files
# ======================================================================

def _parse_dragotto_file(path: pathlib.Path) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Dragotto-format KPG instance file.

    Format:
      Line 1:  n  m
      Line 2:  b_0  b_1  ...  b_{n-1}
      Lines 3+: item_idx  p_0 w_0  p_1 w_1  ...  p_{n-1} w_{n-1}  C_{(0,1)} C_{(0,2)} ... (all ordered pairs)
    """
    with open(path, "r") as f:
        lines = f.readlines()

    n, m = map(int, lines[0].split())
    b = np.array(list(map(float, lines[1].split())), dtype=int)

    p = np.zeros((n, m), dtype=int)
    w = np.zeros((n, m), dtype=int)

    ordered_pairs = list(permutations(range(n), 2))
    C = np.zeros((n, n, m), dtype=int)

    for line in lines[2:]:
        data = list(map(int, line.split()))
        j = data[0]  # item index
        for i in range(n):
            p[i, j] = data[2 * i + 1]
            w[i, j] = data[2 * i + 2]
        offset = 2 * n + 1
        for idx, (a0, a1) in enumerate(ordered_pairs):
            C[a0, a1, j] = data[offset + idx]

    return n, m, p, w, b, C


def load_dragotto_instance(
    path: pathlib.Path,
    *,
    interaction_type: str,
    shared_seed: int = 42,
) -> KPGInstance:
    """Load a Dragotto-format instance and generate shared item capacities.

    Parameters
    ----------
    path : path to .txt file
    interaction_type : "B" or "C"
    shared_seed : seed for generating c_j ~ U{1, n}
    """
    n, m, p, w, b, C = _parse_dragotto_file(path)

    rng = np.random.default_rng(shared_seed)
    c = rng.integers(1, n + 1, size=m, dtype=int)  # c_j ~ U{1,...,n}

    tag = path.stem  # e.g. "2-100-5-cij"
    meta = {
        "source": "dragotto",
        "file": str(path.name),
        "interaction_type": interaction_type,
        "shared_seed": int(shared_seed),
    }

    return KPGInstance(
        n=n, m=m, p=p, w=w, b=b, C=C, c=c,
        interaction_type=interaction_type,
        tag=tag, seed=shared_seed, meta=meta,
    )


def enumerate_dragotto_instances(
    data_dir: pathlib.Path,
    *,
    types: Tuple[str, ...] = ("B", "C"),
    shared_seed: int = 42,
) -> List[KPGInstance]:
    """Enumerate all Type B/C instances from the Dragotto dataset directory.

    Dragotto naming convention:
      - Type A (potential): "*-pot.txt"
      - Type B (positive):  "*-cij.txt"
      - Type C (negative):  "*-cij-n.txt"
    """
    suffix_map = {"B": "-cij.txt", "C": "-cij-n.txt", "A": "-pot.txt", "BC": "-cij-bc.txt"}
    instances = []

    for t in types:
        suffix = suffix_map[t]
        for fp in sorted(data_dir.glob("*.txt")):
            if fp.name.endswith(suffix):
                # Avoid matching "-cij-n.txt" or "-cij-bc.txt" when looking for "-cij.txt"
                if t == "B" and (fp.name.endswith("-cij-n.txt") or fp.name.endswith("-cij-bc.txt")):
                    continue
                inst = load_dragotto_instance(fp, interaction_type=t, shared_seed=shared_seed)
                instances.append(inst)

    return instances


def enumerate_lee_instances(
    data_dir: pathlib.Path,
    *,
    types: Tuple[str, ...] = ("BC",),
    shared_seed: int = 42,
) -> List[KPGInstance]:
    """Enumerate Lee-generated KPG instances (Type BC, etc.).

    Lee naming convention:
      - Type BC (semi-negative): "*-cij-bc.txt"  (C_ij ~ U[-50, 100])
    """
    return enumerate_dragotto_instances(data_dir, types=types, shared_seed=shared_seed)
