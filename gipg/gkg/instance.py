from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import numpy as np

CorrType = Literal["uncorrelated", "weakly_correlated", "strongly_correlated"]


@dataclass(frozen=True)
class GKGInstance:
    """Generalized Knapsack Game instance.

    n players, m items.

    Each player i has:
      - profits p[i,j] >= 0
      - weights w[i,j] >= 0
      - knapsack capacity b[i] >= 0

    Global coupling:
      - item capacities c[j] in {1,...,n}

    All arrays are integer numpy arrays.
    """
    n: int
    m: int
    p: np.ndarray  # shape (n,m)
    w: np.ndarray  # shape (n,m)
    b: np.ndarray  # shape (n,)
    c: np.ndarray  # shape (m,)
    corr: CorrType
    cap_factor: float
    seed: int
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "m": self.m,
            "p": self.p.tolist(),
            "w": self.w.tolist(),
            "b": self.b.tolist(),
            "c": self.c.tolist(),
            "corr": self.corr,
            "cap_factor": float(self.cap_factor),
            "seed": int(self.seed),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GKGInstance":
        return GKGInstance(
            n=int(d["n"]),
            m=int(d["m"]),
            p=np.array(d["p"], dtype=int),
            w=np.array(d["w"], dtype=int),
            b=np.array(d["b"], dtype=int),
            c=np.array(d["c"], dtype=int),
            corr=d["corr"],
            cap_factor=float(d["cap_factor"]),
            seed=int(d.get("seed", 0)),
            meta=dict(d.get("meta", {})),
        )


def _pisinger_like_knapsack(
    rng: np.random.Generator,
    m: int,
    corr: CorrType,
    R: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (weights, profits) with a Pisinger-like scheme.

    This is a standard approximation of Pisinger's generator categories:
      - uncorrelated:  w ~ U[1,R], p ~ U[1,R]
      - weakly correlated: p ~ U[w - R/10, w + R/10]
      - strongly correlated: p = w + R/10

    All values are clipped to be >= 1.

    Note: Duguet et al. cite Silvano et al. (1999) / Pisinger generator.
    If you need exact reproduction, replace this function with an exact port.
    """
    w = rng.integers(1, R + 1, size=m, dtype=int)
    if corr == "uncorrelated":
        p = rng.integers(1, R + 1, size=m, dtype=int)
    elif corr == "weakly_correlated":
        span = R // 10
        low = np.maximum(1, w - span)
        high = w + span
        # randint high is exclusive, so +1
        p = np.array([rng.integers(low[j], high[j] + 1) for j in range(m)], dtype=int)
    elif corr == "strongly_correlated":
        p = w + (R // 10)
    else:
        raise ValueError(f"Unknown corr type: {corr}")
    p = np.maximum(1, p).astype(int)
    return w, p


def generate_gkg_instance(
    *,
    n: int,
    m: int,
    cap_factor: float,
    corr: CorrType,
    seed: int,
    R: int = 1000,
) -> GKGInstance:
    """Generate a GKG instance in the style described by Duguet et al.

    - n in {2,3,4}
    - m in {5,10,15,20,30,40,50}
    - cap_factor in {0.2,0.5} for GNEP knapsack games
    - corr in {uncorrelated, weakly_correlated, strongly_correlated}
    - c[j] uniform in {1,...,n}

    Capacities b_i are set to cap_factor * sum_j w_{ij} and floored to int,
    with a minimum of 1.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros((n, m), dtype=int)
    p = np.zeros((n, m), dtype=int)
    for i in range(n):
        wi, pi = _pisinger_like_knapsack(rng, m, corr, R=R)
        w[i, :] = wi
        p[i, :] = pi

    b = np.maximum(1, np.floor(cap_factor * w.sum(axis=1)).astype(int))
    c = rng.integers(1, n + 1, size=m, dtype=int)  # uniform in {1,...,n}

    meta = {
        "generator": "pisinger_like",
        "R": int(R),
        "notes": "Approximate Pisinger categories; see instance._pisinger_like_knapsack",
    }
    return GKGInstance(n=n, m=m, p=p, w=w, b=b, c=c, corr=corr, cap_factor=cap_factor, seed=seed, meta=meta)
