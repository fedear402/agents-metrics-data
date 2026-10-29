"""Data generator for applied_shift_share.

This is the data step of the task build program (applied-five pilot,
shift_share, function build). It simulates 240 regions and 48 industries
with seed 2026091503 and writes the region-industry rows in a random order.

Usage: python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SEED = 2026091503

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
out_dir = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
nr, ns = 240, 48
industry_mass = np.exp(rng.normal(0,0.5,ns))
alpha = 5*industry_mass/industry_mass.sum()
shares = rng.dirichlet(alpha,nr)
shocks = rng.normal(size=ns)
sector_supply = rng.normal(size=ns)
local_supply, local_noise, treatment_noise = rng.normal(size=(3,nr))
weights = np.exp(rng.normal(0,0.7,nr))
ziv = shares@shocks
exposure_supply = shares@sector_supply
x = 0.4 + 1.4*ziv + 1.5*exposure_supply + 0.7*local_supply + 0.2*treatment_noise
y = 1.0 + 0.8*x + 3*exposure_supply + 0.5*local_supply + 0.4*local_noise
d = pd.DataFrame(dict(region_id=np.repeat(np.arange(1,nr+1),ns),industry_id=np.tile(1000+7*np.arange(ns),nr),share=shares.ravel(),shock=np.tile(shocks,nr),region_weight=np.repeat(weights,ns),x=np.repeat(x,ns),y=np.repeat(y,ns)))
d = d.iloc[rng.permutation(len(d))].reset_index(drop=True)
d.to_csv(out_dir/"input.csv",index=False,float_format="%.17g")
