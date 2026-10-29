"""Generate the input file of lp11_duration.

Copied from the data-generation part of function duration in the task build
program of the likelihood group (seed 170911, NumPy default_rng). Each cohort
draws exact conditional event sets with coefficient 1.1, then censors one
worker after each month's exits.

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

rng=np.random.default_rng(170911);rows=[]
for cohort in range(8):
    training=np.array([0]*9+[1]*9);rng.shuffle(training);alive=list(range(18));exits={}
    for month,d in [(1,5),(2,4),(3,3),(4,2)]:
        combos=list(itertools.combinations(alive,d));weights=np.exp(1.1*np.array([sum(training[list(c)]) for c in combos]));which=rng.choice(len(combos),p=weights/weights.sum());event=combos[which]
        for j in event:exits[j]=(month,1);alive.remove(j)
        if alive:
            j=int(rng.choice(alive));exits[j]=(month,0);alive.remove(j)
    for j in alive:exits[j]=(4,0)
    for j,(end,event) in exits.items():rows.append((cohort,j,int(training[j]),end,event))
df=pd.DataFrame(rows,columns=['cohort','worker','training','end_month','event']).sort_values(['cohort','worker']).reset_index(drop=True)
df.to_csv(OUT / 'input.csv', index=False, float_format='%.15g')
