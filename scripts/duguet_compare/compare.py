"""Merge the sample results: Duguet et al. B&C (same machine + published) vs GZR."""
import pandas as pd, os
H = os.path.join(os.path.dirname(__file__), "sample")
g = pd.read_csv(os.path.join(H, "results_gzr.csv"))
p = pd.read_csv(os.path.join(H, "sample_published.csv"))
m = g.merge(p, on="file", how="left")
bc = os.path.join(H, "results_duguet_bc.csv")
if os.path.exists(bc):
    m = m.merge(pd.read_csv(bc), on="file", how="left")
cols = ["file", "their_status_published", "their_time_published", "their_cuts_published"]
if "their_status" in m: cols += ["their_status", "their_time", "their_cuts"]
cols += ["first_status", "first_time", "first_cuts", "gzr_status", "gzr_total_time", "gzr_cuts", "so_time", "pos"]
pd.set_option("display.width", 250)
print(m[cols].to_string(index=False))
m[cols].to_csv(os.path.join(H, "comparison.csv"), index=False)
