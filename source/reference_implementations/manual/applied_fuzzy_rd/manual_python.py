"""Reference program for applied_fuzzy_rd (Python).

This is the Python reference of the task build program (applied-five pilot,
fuzzy_rd). It fits triangular-weighted local-linear regressions with separate
slopes on the two sides of the cutoff, for earnings and participation, with
NumPy least squares. HC0 influence functions give the joint covariance of the
two jumps, and the delta method gives the ratio SE. The diagnostic values of
the build program are printed as extra keys.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/applied_fuzzy_rd.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/applied_fuzzy_rd")

d=pd.read_csv(INPUT / "input.csv"); d=d[np.abs(d.score)<1]
x=d.score.to_numpy(); a=(x>=0).astype(float)
X=np.column_stack([np.ones(len(x)),a,x,a*x]); w=1-np.abs(x)
Y=d[["earnings","participated"]].to_numpy()
B=np.linalg.lstsq(X*np.sqrt(w[:,None]),Y*np.sqrt(w[:,None]),rcond=None)[0]
U=Y-X@B; bread=np.linalg.inv(X.T@(w[:,None]*X))
influence=(X@bread[:,1]*w)[:,None]*U
C=influence.T@influence
rf,fs=B[1]; tau=rf/fs; grad=np.array([1/fs,-rf/fs**2])
se=np.array([np.sqrt(grad@C@grad),np.sqrt(C[0,0]),np.sqrt(C[1,1])])
out=dict(estimates=[dict(term=t,estimate=float(v),se=float(s),n_obs=len(x)) for t,v,s in zip(["treatment_effect","reduced_form","first_stage"],[tau,rf,fs],se)],
 covariance=C.tolist(),ratio_se_without_cross_covariance=float(np.sqrt(np.sum(grad**2*np.diag(C)))))
print(json.dumps(out))
