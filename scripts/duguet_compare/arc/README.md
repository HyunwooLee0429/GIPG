# Full Duguet-vs-GZR run on one ARC node

One named-user Gurobi license per node (bound to its hostname); all parallelism
is processes on that node. Their code caps Gurobi at one thread
(`ThreadLimit=1` in `BranchAndCutForGNEP.cpp`), so GZR is run with `--threads 1`
for the like-for-like comparison.

```
# login node: clone both repos, then
bash scripts/duguet_compare/arc/setup_node.sh ~/branch-and-cut-for-ipgs   # build (needs internet: Catch2 fetch)

# get a whole node
salloc --nodes=1 --exclusive --time=1-00:00:00 --account=<alloc>
srun --pty bash            # or ssh to the node salloc reports
hostname                   # -> request the Gurobi key for THIS host, then:
grbgetkey <key>            # writes ~/gurobi.lic
bash scripts/duguet_compare/arc/setup_node.sh ~/branch-and-cut-for-ipgs   # re-runs only the license check
bash scripts/duguet_compare/arc/run_all_parallel.sh ~/branch-and-cut-for-ipgs 600 32
```

Budget at a 600 s limit, 32 parallel B&C jobs: their side ≤ 1,260·600/32 ≈ 6.6 h
worst case (about 3–4 h expected); GZR side < 1 h. At their original 3,600 s
limit: ≤ 40 h worst case, so ask for `--time=2-00:00:00` (long QoS).

Outputs in `scripts/duguet_compare/full/` (git-ignored):
- `bc/<instance>.result|.log` — their one-line result and verbosity-1 log per instance
- `gzr_logs/<instance>/` — Gurobi logs (gzr_master, first_master, so), `cuts.jsonl`, `provenance.json`
- `results_duguet_bc.csv`, `results_gzr.csv`, `comparison_full.csv`, `run_log.txt`

Everything is resume-safe: if the allocation ends, get a new node (same hostname
via `--nodelist`, or a new key for the new host) and rerun the same command.
If a compute node has no outbound internet, `grbgetkey` will fail there — check with
`curl -sI https://portal.gurobi.com | head -1` first; the fallback is the ISE
license server or a desktop with a named-user license.
