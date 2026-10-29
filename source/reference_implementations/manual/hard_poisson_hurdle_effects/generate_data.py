"""Generate the input of hard_poisson_hurdle_effects.

This is the data step of the task build program (function hurdle).
Usage:
    python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv with the shipped file name.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091703);n=1300;x=rng.normal(scale=.7,size=n);d=rng.binomial(1,.45,size=n);X=np.column_stack([np.ones(n),x,d]);prob=expit(-.3+.6*x-.4*d);lam=np.exp(.1+.45*x+.35*d)
positive=rng.binomial(1,prob).astype(bool);y=np.zeros(n,dtype=int)
# Positive rows draw from Poisson(lam) until the draw is positive (zero truncation).
for i in np.flatnonzero(positive):
    while y[i]==0:y[i]=rng.poisson(lam[i])
frame=pd.DataFrame(dict(y=y,x=x,d=d))
frame.to_csv(OUT/'input.csv',index=False,float_format='%.17g')
