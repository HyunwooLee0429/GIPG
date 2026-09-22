"""Parse Duguet et al. GNEP knapsack instance files (N_m_c_corr_k.txt) into GKGInstance.

File format (instances/README.md of github.com/AloisDuguet/branch-and-cut-for-ipgs):
  line 1: N m
  line 2: b_1 ... b_N
  item lines: j  p_1j w_1j ... p_Nj w_Nj  C_{1,2,j} ... C_{N,N-1,j}  c_j
The C interaction coefficients are ignored by their GNEP-fullInteger build
(createGNEPKnapsackInstance sets Q = {}), so the game is linear GKG.
"""
from __future__ import annotations
import os, re
import numpy as np
from gipg.gkg.instance import GKGInstance

_CORR = {"uncorr": "uncorrelated", "weakcorr": "weakly_correlated", "strongcorr": "strongly_correlated"}


def parse_duguet_gkg(path: str) -> GKGInstance:
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    n, m = map(int, lines[0].split())
    b = np.array(list(map(int, lines[1].split()))[:n], dtype=int)
    p = np.zeros((n, m), dtype=int); w = np.zeros((n, m), dtype=int); c = np.zeros(m, dtype=int)
    for line in lines[2:2 + m]:
        t = list(map(int, line.split()))
        j = t[0]
        for i in range(n):
            p[i, j] = t[1 + 2 * i]; w[i, j] = t[2 + 2 * i]
        # after 2n p/w entries come n(n-1) interaction coefficients, then c_j
        expected = 1 + 2 * n + n * (n - 1)
        if len(t) == expected + 1:
            c[j] = t[-1]
        else:
            raise ValueError(f"{path}: item line {j} has {len(t)} fields, expected {expected + 1}")
    base = os.path.basename(path)
    mt = re.match(r"^(\d+)-(\d+)-(\d+)-(\w+?)-(\d+)\.txt$", base)
    cf = int(mt.group(3)) / 10.0
    corr = _CORR[mt.group(4)]
    # sanity: b_i = floor(cf * sum_j w_ij) in their generator

    return GKGInstance(n=n, m=m, p=p, w=w, b=b, c=c, corr=corr, cap_factor=cf,
                       seed=int(mt.group(5)), meta={"source": "duguet2025", "file": base})


if __name__ == "__main__":
    import sys
    inst = parse_duguet_gkg(sys.argv[1])
    print(inst.n, inst.m, inst.corr, inst.cap_factor, "b=", inst.b.tolist(), "c=", inst.c.tolist())
    print("b == floor(cf*sum w):", (inst.b == np.floor(inst.cap_factor * inst.w.sum(axis=1)).astype(int)).all())
