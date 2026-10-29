"""Generate the input file of lp04_ppml.

Copied from the data-generation part of function ppml in the task build
program of the likelihood group (seed 170904, NumPy default_rng). The program
plants one zero-only pair and one zero-only exporter-year.

Usage:
    python generate_data.py OUT_DIR
Writes input.csv into OUT_DIR.
"""
from pathlib import Path
import itertools
import sys

import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(170904)
rows=[]
for i,j,t in itertools.product(range(5),range(5),range(6)):
    policy=int(t>=2+(i+j)%3 and (i*2+j)%3!=0)
    mean=np.exp(.6+.65*policy+.3*np.sin(i+j)+.15*t-.12*i+.1*j)
    y=int(rng.poisson(mean))
    if (i==0 and j==0) or (i==4 and t==0):y=0
    rows.append((i,j,t,f'{i}_{j}',f'{i}_{t}',f'{j}_{t}',policy,y))
df=pd.DataFrame(rows,columns=['exporter','importer','year','pair','exporter_year','importer_year','agreement','trade'])
df.to_csv(OUT / 'input.csv', index=False, float_format='%.15g')
