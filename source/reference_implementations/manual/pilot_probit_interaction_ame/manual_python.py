"""Reference program for pilot_probit_interaction_ame (Python).

This is the main route of the task build program (function nonlinear):
statsmodels Probit on the full sample, then the analytic average marginal
effect of x and the average discrete effect of d over the target rows.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/pilot_probit_interaction_ame. It must hold input.csv.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import ndtr
from scipy.stats import norm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/pilot_probit_interaction_ame")

frame=pd.read_csv(INPUT / 'input.csv')
y=frame.y.to_numpy();x=frame.x.to_numpy();d=frame.d.to_numpy();z=frame.z.to_numpy();group=frame.target.to_numpy();n=len(frame)
X=np.column_stack([np.ones(n),x,d,x*d,z,x*x])
fit=sm.Probit(y,X).fit(disp=False,tol=1e-11,maxiter=300)
b=fit.params;mask=group==1
p0=np.column_stack([np.ones(n),x,np.zeros(n),np.zeros(n),z,x*x])
p1=p0.copy();p1[:,2]=1;p1[:,3]=x
result={'ame_x':float(np.mean(norm.pdf(X[mask]@b)*(b[1]+b[3]*d[mask]+2*b[5]*x[mask]))),
        'effect_d':float(np.mean(ndtr(p1[mask]@b)-ndtr(p0[mask]@b)))}

print(json.dumps({
    "task_id": "pilot_probit_interaction_ame",
    "estimates": [{"term": k, "estimate": v, "se": None, "n_obs": int(n)} for k, v in result.items()],
}))
