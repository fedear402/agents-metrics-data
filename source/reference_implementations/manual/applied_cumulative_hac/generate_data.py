"""Data generator for applied_cumulative_hac.

This is the data step of the task build program (applied-five pilot,
cumulative_hac, function build). It simulates 720 consecutive weekday
observations with seed 2026091502.

Usage: python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
out_dir = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091502); n=720
lv=np.zeros(n); lv[0]=np.log(20); u=np.zeros(n)
for t in range(1,n):
    lv[t]=np.log(20)+.85*(lv[t-1]-np.log(20))+.06*rng.normal()
    u[t]=.75*u[t-1]+.05*rng.normal()
delta=np.r_[0,np.diff(lv)]
beta=.6*.85**np.arange(15); beta[-1]=-.2
ls=np.full(n,4.0)+u
for j in range(1,16): ls[j:]+=beta[j-1]*delta[:-j]
path=out_dir/'input.csv'
pd.DataFrame(dict(date=pd.bdate_range('2019-01-02',periods=n).strftime('%Y-%m-%d'),sales=np.exp(ls),vix=np.exp(lv))).to_csv(path,index=False,float_format='%.17g')
