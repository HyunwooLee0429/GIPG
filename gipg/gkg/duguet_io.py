"""Reader for the GNEP knapsack instance files of Duguet et al.

File layout, as documented in ``instances/README.md`` of their repository::

    N m
    b_1 ... b_N
    j  p_1j w_1j ... p_Nj w_Nj  C_1,2,j ... C_N,N-1,j  c_j

one item line per item. The interaction coefficients ``C`` are present in the
file but are not part of the generalized knapsack game: in that game a player's
payoff depends only on her own selections, and the players are coupled solely
through the item availabilities ``c_j``. They are read and returned in ``meta``
so that the file can be round-tripped, but they are not used.

The file name encodes the instance parameters as ``N-m-c-corr-k.txt`` with
``corr`` in ``{uncorr, weakcorr, strongcorr}`` and ``k`` a replication counter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional
import re

import numpy as np

from .instance import GKGInstance

_CORR = {
    "uncorr": "uncorrelated",
    "weakcorr": "weakly_correlated",
    "strongcorr": "strongly_correlated",
}

_NAME = re.compile(r"^(\d+)-(\d+)-(\d+)-(uncorr|weakcorr|strongcorr)-(\d+)$")


def _parse_name(stem: str):
    mt = _NAME.match(stem)
    if mt is None:
        return None
    n, m, cap10, corr, rep = mt.groups()
    return int(n), int(m), int(cap10) / 10.0, _CORR[corr], int(rep)


def read_instance(path: str | Path) -> GKGInstance:
    """Read one GNEP knapsack instance file into a :class:`GKGInstance`."""
    path = Path(path)
    tokens = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]

    n, m = (int(t) for t in tokens[0])
    b = np.array([int(t) for t in tokens[1]], dtype=int)
    if b.shape != (n,):
        raise ValueError(f"{path.name}: expected {n} capacities, got {b.shape[0]}")
    if len(tokens) - 2 != m:
        raise ValueError(f"{path.name}: expected {m} item lines, got {len(tokens) - 2}")

    width = 1 + 2 * n + n * (n - 1) + 1
    p = np.zeros((n, m), dtype=int)
    w = np.zeros((n, m), dtype=int)
    c = np.zeros(m, dtype=int)
    interactions: List[List[int]] = []

    for row in tokens[2:]:
        if len(row) != width:
            raise ValueError(
                f"{path.name}: item line has {len(row)} fields, expected {width}"
            )
        vals = [int(t) for t in row]
        j = vals[0]
        if not 0 <= j < m:
            raise ValueError(f"{path.name}: item index {j} out of range")
        pw = vals[1 : 1 + 2 * n]
        p[:, j] = pw[0::2]
        w[:, j] = pw[1::2]
        interactions.append(vals[1 + 2 * n : -1])
        c[j] = vals[-1]

    meta = {"file": path.name, "source": "duguet", "interactions": interactions}
    parsed = _parse_name(path.stem)
    if parsed is None:
        corr, cap_factor, rep = "uncorrelated", 0.0, 0
    else:
        n_name, m_name, cap_factor, corr, rep = parsed
        if (n_name, m_name) != (n, m):
            raise ValueError(
                f"{path.name}: name says n={n_name}, m={m_name}; file says n={n}, m={m}"
            )
    meta["replication"] = rep

    return GKGInstance(
        n=n, m=m, p=p, w=w, b=b, c=c,
        corr=corr, cap_factor=cap_factor, seed=rep, meta=meta,
    )


def read_directory(root: str | Path) -> Iterator[GKGInstance]:
    """Read every ``*.txt`` instance under ``root``, in sorted file order."""
    for path in sorted(Path(root).glob("*.txt")):
        yield read_instance(path)


def check_instance(inst: GKGInstance, tol: int = 1) -> List[str]:
    """Return a list of consistency complaints; empty means the file looks sane.

    Checks the invariants the generator of Duguet et al. is documented to
    satisfy: positive profits and weights, item availabilities in ``1..n``, and
    strongly correlated profits equal to the weight plus ``R/10``.
    """
    out: List[str] = []
    if inst.p.min() < 1 or inst.w.min() < 1:
        out.append("non-positive profit or weight")
    if inst.c.min() < 1 or inst.c.max() > inst.n:
        out.append(f"item availability outside 1..{inst.n}")
    if inst.corr == "strongly_correlated":
        offsets = np.unique(inst.p - inst.w)
        if offsets.size != 1:
            out.append(f"strong correlation not constant: offsets {offsets.tolist()}")
    return out
