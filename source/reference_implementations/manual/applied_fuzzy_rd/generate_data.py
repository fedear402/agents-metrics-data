"""Data generator for applied_fuzzy_rd.

This is the data step of the task build program (applied-five pilot,
fuzzy_rd, function build). It simulates 1,200 applicants with seed
2026091501.

Usage: python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtri

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
out_dir = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091501); n=1200
x=rng.uniform(-2,2,n); u=rng.normal(size=n); v=rng.normal(size=n)
p=.2+.5*(x>=0)+.04*x
take=(u<ndtri(p)).astype(int)
y=2+1.5*take+.3*x+.4*(x>=0)*x+1.5*u+.6*v
path=out_dir/'input.csv'
pd.DataFrame(dict(id=np.arange(1,n+1),score=x,participated=take,earnings=y)).to_csv(path,index=False,float_format='%.17g')
