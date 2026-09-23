# GIPG — Pure Nash equilibria of generalized integer programming games

Reference implementation for the paper *Conditional Equilibrium Inequalities for
Generalized Integer Programming Games*.

A **generalized integer programming game** (GIPG) is a non-cooperative game in
which every player solves an integer program and shared constraints couple the
players' feasible sets, so that a deviation which is privately feasible may be
jointly infeasible. This repository implements **conditional equilibrium
inequalities** (CEIs) — globally valid linear inequalities that enforce the
no-regret condition only when a deviation is actually feasible — together with
the three algorithms built on them and the four game classes studied in the
paper.

## Algorithms

| Name | Function | Purpose |
| --- | --- | --- |
| GZR | `solve_gzr` | Generalized Zero-Regret. Separates CEIs as lazy constraints inside a branch-and-cut MILP and optimizes an arbitrary linear objective over the set of pure Nash equilibria, yielding the socially optimal PNE and the exact price of stability. |
| RRR-BRD | `brd_random_restart` | Random-restart best-response dynamics. Fast heuristic; supplies a warm start for GZR. |
| ABS | `solve_abs` | Alpha-bisection search. Brackets the tightest `alpha` admitting an `alpha`-approximate PNE, reusing CEIs across iterations. |

## Game classes

| Module | Game | Coupling |
| --- | --- | --- |
| `gipg.isbp` | Integer splittable bin packing | Shared bin capacities; proportional cost sharing |
| `gipg.nfg` | Network formation with integer-splittable routing | Shared edge capacities; proportional cost sharing |
| `gipg.gkg` | Generalized knapsack | Per-item capacities shared across players |
| `gipg.gkgb` | Generalized knapsack, reciprocally bilinear payoffs | As above, plus bilinear interaction terms |

## Installation

Requires Python 3.10+ and a working Gurobi installation with a licence
(academic licences are free).

```bash
git clone https://github.com/HyunwooLee0429/GIPG.git
cd GIPG
pip install -e .
```

For the notebooks:

```bash
pip install -e ".[notebooks]"
```

## Usage

Every game module exposes the same interface.

```python
from gipg import nfg

inst = nfg.generate_instance(
    n_players=4,
    graph_type="layered",
    n_layers=5,
    n_width=5,
    capacity_level="tight",
    seed=0,
)

# Fast heuristic: returns a profile and its approximation ratio
brd = nfg.brd_random_restart(inst, max_init=3, max_round=15)

# Socially optimal PNE
res = nfg.solve_gzr(inst, alpha=1.0, time_limit=600.0)
print(res.status, res.obj_val)

# Price of stability
so = nfg.solve_social_optimum(inst, time_limit=1200.0)
print(nfg.compute_pos(res.obj_val, so.opt_cost))
```

If no exact equilibrium exists, bracket the tightest approximate one:

```python
abs_res = nfg.solve_abs(inst, eps=1e-2)
print(abs_res.alpha_lb_verified, abs_res.alpha_ub_verified)
```

The same calls work for `gipg.isbp`, `gipg.gkg` and `gipg.gkgb`; only the
instance constructor differs.

## Repository layout

```
gipg/           package, one subpackage per game class
  isbp/ nfg/ gkg/ gkgb/
    instance.py        instance representation and generators
    objectives.py      cost / welfare evaluation
    best_response.py   single-player best-response oracle
    cuts.py            CEI construction
    duguet_io.py       reader for the instance files of Duguet et al. (gkg only)
    model.py           master MILP  (isbp: folded into gzr.py)
    gzr.py             GZR branch-and-cut loop
    abs_search.py      alpha-bisection search
    heuristics.py      RRR-BRD and profile utilities
    social_optimum.py  social optimum and price of stability
notebooks/      one notebook per experiment
data/           benchmark instances
results/        result CSVs behind the tables in the paper
scripts/        table generation and the comparison against Duguet et al.
```

## Reproducing the experiments

Instances for NFG, GKGB and the GKG scalability study are shipped in `data/`;
ISBP instances are generated deterministically from `(n, m, w, u, seed)` and
need no data files, and the GKG benchmark uses the instance files of Duguet
et al. (see below). Run the notebook for a game and it writes its CSV into
`results/`.

| Notebook | Produces | Tables |
| --- | --- | --- |
| `isbp_regular.ipynb` | `isbp_exp1_regular.csv`, `isbp_exp2_abs.csv` | ISBP summary, ABS, comprehensive (regular) |
| `isbp_nonregular.ipynb` | `isbp_exp1_nonregular.csv` | ISBP summary, comprehensive (non-regular) |
| `isbp_regular_no_vest.ipynb`, `isbp_nonregular_no_vest.ipynb` | `*_no_vest.csv` | VEST cut comparison |
| `nfg_generate_instances.ipynb` | `data/nfg/*.json` | — |
| `nfg_run.ipynb`, `nfg_run_no_vest.ipynb` | `nfg_exp1.csv`, `nfg_exp1_no_vest.csv` | NFG summary, by capacity and players, comprehensive, VEST comparison |
| `nfg_recompute_social_optimum.ipynb` | `nfg_social_optimum.csv` | recomputes the social-optimum reference at a 1,200 s limit |
| `gkg_run_scalability.ipynb` | `gkg_exp1_scalability.csv` | GKG summary, comprehensive |
| `gkgb_run_typeC.ipynb`, `gkgb_run_typeBC.ipynb` | `gkgb_exp1_*.csv`, `gkgb_exp2_*.csv` | GKGB summary, ABS, comprehensive |

`scripts/make_nfg_tables.py` regenerates every NFG table from the two NFG result
CSVs, so those tables cannot drift apart:

```bash
python scripts/make_nfg_tables.py          # writes results/nfg_tables.tex
```

## Comparison with Duguet et al.

The GKG benchmark row of the paper is run on the 1,260 generalized knapsack
instances published by Duguet et al., not on instances of our own, so that both
methods are evaluated on identical data. Their instances and their raw results
are not redistributed here; clone their repository and point the scripts at it.

```bash
git clone https://github.com/AloisDuguet/branch-and-cut-for-ipgs.git

# our runs: 16 threads, and single-threaded to match their single-core setup
python scripts/run_duguet_benchmark.py \
    branch-and-cut-for-ipgs/instances/GNEP_knapsack_instances \
    --out results/gkg_duguet_instances.csv
python scripts/run_duguet_benchmark.py \
    branch-and-cut-for-ipgs/instances/GNEP_knapsack_instances \
    --out results/gkg_duguet_1thread.csv --threads 1 --skip-optimal

# per-cell comparison table
python scripts/compare_duguet.py --ours results/gkg_duguet_1thread.csv \
    --theirs branch-and-cut-for-ipgs/results/GNEPKnapsack/results.csv
```

`run_duguet_benchmark.py` runs two configurations per instance. `find` gives GZR
no warm start and stops at the first equilibrium, matching the task their
branch-and-cut solves; `optimal` warm-starts with RRR-BRD and optimizes to gap
zero, returning the socially optimal equilibrium and the price of stability,
which their method does not address. `gipg.gkg.duguet_io` reads their instance
file format.

## Solver settings

The master problem uses `LazyConstraints=1`; the social-optimum model, which
needs no separation, uses `LazyConstraints=0`. All numerical tolerances are left
at Gurobi defaults (`MIPGap=1e-4`, `IntFeasTol=1e-5`, `FeasibilityTol=1e-6`) for
both the master and the best-response subproblems, so a reported status of
`OPTIMAL` certifies optimality within the default relative gap rather than in
exact arithmetic. A CEI violation is declared when the incumbent's regret exceeds
`1e-9`. Both the master and the best-response models run with 16 threads by
default; pass `threads=` to change this. Every result in the paper uses 16
threads except the comparison against Duguet et al., whose experiments use a
single core: that benchmark is rerun with `--threads 1`, which also sets the
best-response threads. The setting makes little difference here — 0.066 s mean
against 0.072 s with 16 threads.

The VEST cut and, for ISBP, the player-anchoring constraint are valid for exact
equilibria only. `solve_abs` therefore imposes them just in its initial test at
`alpha = 1` and drops them in every bisection call above it: with either active,
an infeasible outcome at `alpha > 1` would not certify that no `alpha`-PNE
exists. Bin-load ordering is valid at every `alpha` and stays on throughout.
Anchoring is selected automatically on regular ISBP instances and is imposed only
where `w <= u`, the range in which it is proved; passing `add_player_anchoring`
explicitly overrides the automatic choice.

Results in the paper were produced with Gurobi 13.0 on an Intel Core Ultra 7
265F with 32 GB RAM.

## Citation

```bibtex
@article{gipg,
  title  = {Conditional Equilibrium Inequalities for Generalized Integer Programming Games},
  author = {Lee, Hyunwoo and Hildebrand, Robert and Michini, Carla and Hao, Bainian},
  year   = {2026}
}
```

## License

To be determined.
