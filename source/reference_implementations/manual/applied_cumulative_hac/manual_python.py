"""Reference program for applied_cumulative_hac (Python).

This is the Python reference of the task build program (applied-five pilot,
cumulative_hac). It fits OLS of log sales on lags 1 to 15 of the log-vix
change and uses the statsmodels Bartlett HAC covariance with 15 lags and the
N/(N-K) correction. The diagnostic values of the build program are printed as
extra keys.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/applied_cumulative_hac.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/applied_cumulative_hac")

d=pd.read_csv(INPUT / "input.csv").sort_values("date")
dx=np.log(d.vix).diff()
lags=pd.concat([dx.shift(j).rename("lag_"+str(j)) for j in range(1,16)],axis=1)
valid=lags.notna().all(axis=1)
X=sm.add_constant(lags[valid]).to_numpy(); y=np.log(d.sales[valid]).to_numpy()
fit=sm.OLS(y,X).fit()
hac=fit.get_robustcov_results(cov_type="HAC",maxlags=15,use_correction=True)
V=hac.cov_params(); b=fit.params
C=np.zeros((3,16)); C[0,1:]=1; C[1,1]=1; C[2,15]=1
values=C@b; se=np.sqrt(np.diag(C@V@C.T))
hc0=fit.get_robustcov_results(cov_type="HC0").cov_params()
out=dict(estimates=[dict(term=t,estimate=float(v),se=float(s),n_obs=len(y)) for t,v,s in zip(["cumulative_response","lag_1","lag_15"],values,se)],
 cumulative_se_hc0=float(np.sqrt(C[0]@hc0@C[0])),cumulative_se_ignoring_covariance=float(np.sqrt(np.diag(V)[1:].sum())),
 covariance=V.tolist(),sample_start=str(d.loc[valid,"date"].iloc[0]),sample_end=str(d.loc[valid,"date"].iloc[-1]))
print(json.dumps(out))
