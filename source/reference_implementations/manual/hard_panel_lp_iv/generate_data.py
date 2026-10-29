"""Generate the input file of hard_panel_lp_iv.

Copied from the data-generation part of function panel_lp_iv in the task
build program of the IV and time-series group (seed 170706, NumPy
default_rng; row order shuffled with pandas random_state 12).

Usage:
    python generate_data.py OUT_DIR
Writes input.csv into OUT_DIR.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(170706); rows=[]
for i in range(25):
    y=0.;a=rng.normal()
    for t in range(35):
        z=rng.normal();v=rng.normal();x=.8*z+.7*v;y=a+.5*y+.9*x+.5*v+rng.normal(scale=.4)
        if rng.random()>.12:rows.append((i,t,y,x,z))
d=pd.DataFrame(rows,columns=['unit','period','y','x','z']).sample(frac=1,random_state=12)
d.to_csv(OUT/'input.csv',index=False,float_format='%.17g')
