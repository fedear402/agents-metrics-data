"""Reference program for broad_multinomial_logit (Python).

This is the Python reference program of the task build. It fits a multinomial
logit with base outcome 0 by Newton steps (statsmodels MNLogit) and reports
observed-information standard errors and the fitted probability of outcome 2
at x=0 and z=0.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/broad_multinomial_logit.
The program prints one JSON object and writes no files.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/broad_multinomial_logit")
d=pd.read_csv(INPUT / "input.csv")
rows=[]
def add(term,estimate,se=None,n=None):
    rows.append(dict(term=term,estimate=float(estimate),se=None if se is None else float(se),n_obs=int(len(d) if n is None else n)))
f=sm.MNLogit(d.y,sm.add_constant(d[["x","z"]])).fit(method="newton",disp=False,tol=1e-12,maxiter=1000)
for k in [1,2]:
    for var in ["x","z"]: add(f"{var}_cat{k}",f.params.loc[var,k-1],f.bse.loc[var,k-1])
add("prob_cat2_zero",f.predict(np.array([[1.,0.,0.]]))[0,2])
print(json.dumps({"estimates":rows},indent=2))
