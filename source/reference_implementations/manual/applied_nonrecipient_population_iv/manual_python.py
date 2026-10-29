"""Reference program for applied_nonrecipient_population_iv (Python).

This is the reference route of the task build program (function c2): one
interacted weighted 2SLS system by projection, then group effects and the
census-weighted population effect.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/applied_nonrecipient_population_iv. It must hold
households.csv and census.csv.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/applied_nonrecipient_population_iv")


def iv(X, Z, y, w):
    """Weighted projection 2SLS, including all exogenous columns in Z."""
    s = np.sqrt(w)
    a, b, c = X*s[:,None], Z*s[:,None], y*s
    fitted = b @ np.linalg.lstsq(b, a, rcond=None)[0]
    return np.linalg.solve(fitted.T @ a, fitted.T @ c)


h=pd.read_csv(INPUT / 'households.csv')
c=pd.read_csv(INPUT / 'census.csv')
d=h[h.recipient==0].copy()


def compute(frame):
    e=frame.eligible.to_numpy(); x=frame.dose.to_numpy(); z=frame.instrument.to_numpy(); a=frame.baseline.to_numpy(); w=frame.weight.to_numpy(); y=frame.outcome.to_numpy()
    C=np.column_stack([1-e,e,a*(1-e),a*e]); X=np.column_stack([C,x*(1-e),x*e]); Z=np.column_stack([C,z*(1-e),z*e])
    betas=iv(X,Z,y,w)[-2:]
    effects=[float(betas[g]*np.average(x[e==g],weights=w[e==g])) for g in (0,1)]
    pop=c.sort_values('eligible').nonrecipient_population.to_numpy()
    return {'ineligible_effect':effects[0],'eligible_nonrecipient_effect':effects[1],'population_effect':float(np.average(effects,weights=pop))}


main=compute(d)
n=len(d)

print(json.dumps({
    "task_id": "applied_nonrecipient_population_iv",
    "estimates": [{"term": k, "estimate": v, "se": None, "n_obs": n} for k, v in main.items()],
}))
