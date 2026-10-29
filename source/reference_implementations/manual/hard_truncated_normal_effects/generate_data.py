"""Generate the input of hard_truncated_normal_effects.

This is the data step of the task build program (function truncated).
Usage:
    python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv with the shipped file name.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091700);n=1800;x=rng.normal(size=n);z=rng.normal(size=n)
y=-.45+.8*x-.3*z+rng.normal(scale=1.1,size=n)
keep=y>0;d=pd.DataFrame(dict(y=y[keep],x=x[keep],z=z[keep]))
d.to_csv(OUT/'input.csv',index=False,float_format='%.17g')
