"""Generate the input of broad_cohort_stack.

This is the data step of the task build program (seed 2026091562).
It draws 160 units: 20 with adoption period 3, 40 with 5, 60 with 7, and 40
never treated (g = 0). Each unit-time row is dropped with probability 0.065.
Usage:
    python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv with the shipped file name.
It prints the SHA-256 of the file and compares it with the shipped file.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHA256 = "32c59e41ffcf9b70bb68bd8256b94ae525d84c9e4a91f047ad77381245f51f9e"

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091562);rows=[]
for unit in range(1,161):
    g=[3,5,7,0][np.searchsorted([20,60,120],unit-1,side='right')];a=rng.normal();shock=rng.normal(size=10)
    for t in range(1,11):
        if rng.random()<.065:continue
        tau=(.5+.15*g+.1*(t-g)) if g>0 and t>=g else 0
        rows.append(dict(id=unit,time=t,g=g,y=a+.2*t+.25*np.sin(t)+tau+shock[t-1]))
data=pd.DataFrame(rows)
data.to_csv(OUT/'input.csv',index=False,float_format='%.17g')

digest = hashlib.sha256((OUT / "input.csv").read_bytes()).hexdigest()
print("input.csv", digest, "matches the shipped file" if digest == EXPECTED_SHA256 else "DIFFERS from the shipped file")
