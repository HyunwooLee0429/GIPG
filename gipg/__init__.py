"""Computing pure Nash equilibria of generalized integer programming games.

A generalized integer programming game (GIPG) is a non-cooperative game in which
each player solves an integer program and shared constraints couple the players'
feasible sets. This package implements conditional equilibrium inequalities
(CEIs) and the algorithms built on them, together with the four game classes
used in the accompanying paper.

Each game module exposes the same interface::

    from gipg import nfg

    inst = nfg.generate_instance(...)
    res  = nfg.solve_gzr(inst, alpha=1.0)
    tight = nfg.solve_abs(inst)

Modules
-------
isbp
    Integer splittable bin packing games.
nfg
    Network formation games with integer-splittable routing.
gkg
    Generalized knapsack games.
gkgb
    Generalized knapsack games with reciprocally bilinear payoffs.
"""

from . import isbp, gkg, gkgb, nfg

__all__ = ["isbp", "gkg", "gkgb", "nfg"]
__version__ = "1.0.0"
