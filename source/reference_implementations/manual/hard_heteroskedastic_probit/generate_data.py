"""Generate the input of hard_heteroskedastic_probit.

This is the data step of the task build program (function heteroprobit).
Usage:
    python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv with the shipped file name.
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

rng=np.random.default_rng(2026091702);n=1400;x=rng.normal(size=n);z=rng.normal(scale=.8,size=n);d=rng.binomial(1,.5,size=n)
X=np.column_stack([np.ones(n),x,d]);eta=(.15+.7*x-.55*d)/np.exp(.65*z);y=rng.binomial(1,ndtr(eta))
frame=pd.DataFrame(dict(y=y,x=x,d=d,z=z))
frame.to_csv(OUT/'input.csv',index=False,float_format='%.17g')
