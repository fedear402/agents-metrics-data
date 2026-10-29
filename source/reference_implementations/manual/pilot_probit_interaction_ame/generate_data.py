"""Generate the input file of pilot_probit_interaction_ame.

Copied from the data-generation part of function nonlinear in the build
program of the three difficulty pilots (seed 91703, NumPy default_rng, 850
rows).

Usage:
    python generate_data.py OUT_DIR
Writes input.csv into OUT_DIR.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(91703);n=850
x=rng.normal(size=n);z=rng.normal(size=n);d=rng.binomial(1,ndtr(.5*x))
eta=-.25+.45*x+.65*d-.75*x*d+.28*z+.18*x*x
y=rng.binomial(1,ndtr(eta));group=(x+.4*z>-.3).astype(int)
frame=pd.DataFrame(dict(y=y,x=x,d=d,z=z,target=group))
frame.to_csv(OUT / 'input.csv', index=False, float_format='%.17g')
