"""Reference program for broad_ordered_logit (Python).

This is the Python reference program of the task build. It fits an ordered
logit with two free cutpoints and no intercept (statsmodels OrderedModel, BFGS)
and reports observed-information standard errors and the fitted probability of
category 3 at x=1 and z=0.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/broad_ordered_logit.
The program prints one JSON object and writes no files.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/broad_ordered_logit")
d=pd.read_csv(INPUT / "input.csv")
rows=[]
def add(term,estimate,se=None,n=None):
    rows.append(dict(term=term,estimate=float(estimate),se=None if se is None else float(se),n_obs=int(len(d) if n is None else n)))
from statsmodels.miscmodels.ordinal_model import OrderedModel
f=OrderedModel(d.y,d[["x","z"]],distr="logit").fit(method="bfgs",disp=False,gtol=1e-10,maxiter=1000)
add("x",f.params.iloc[0],f.bse.iloc[0]);add("z",f.params.iloc[1],f.bse.iloc[1])
add("prob_high",f.model.predict(f.params,exog=np.array([[1.,0.]]))[0,2])
print(json.dumps({"estimates":rows},indent=2))
