"""Data generator for hard_conditional_logit_welfare.

This is the data step of the task build program (twenty-harder-tasks,
nonlinear group, function choice_welfare). It simulates 550 people with three
alternatives each, with seed 2026091704.

Usage: python generate_data.py OUT_DIR
The program writes OUT_DIR/input.csv.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

if len(sys.argv) != 2:
    sys.exit("usage: python generate_data.py OUT_DIR")
out_dir = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2026091704);n=550;J=3
cost=rng.uniform(.4,3.5,size=(n,J));quality=rng.normal(size=(n,J));asc=np.tile(np.eye(J)[None,:,:],(n,1,1))[:,:,1:]
X=np.concatenate([cost[:,:,None],quality[:,:,None],asc],axis=2);truth=np.array([-1.0,.65,.4,-.2]);utility=X@truth;pr=np.exp(utility-logsumexp(utility,axis=1)[:,None]);choice=np.array([rng.choice(J,p=p) for p in pr]);w=rng.uniform(.5,2.5,n)
d=pd.DataFrame(dict(person=np.repeat(np.arange(n),J),alternative=np.tile(np.arange(J),n),chosen=(np.tile(np.arange(J),(n,1))==choice[:,None]).reshape(-1).astype(int),cost=cost.reshape(-1),quality=quality.reshape(-1),weight=np.repeat(w,J)))
d.to_csv(out_dir/'input.csv',index=False,float_format='%.17g')
