"""Reference program for lp11_duration (Python).

This is the reference route of the task build program (function duration):
cohort-month risk sets, the exact conditional log likelihood through a
log-space elementary symmetric polynomial recursion, and bounded scalar
maximization over beta in [-4, 4].

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/lp11_duration. It must hold input.csv.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/lp11_duration")

df=pd.read_csv(INPUT / 'input.csv')
sets=[]
for c in range(8):
    z=df[df.cohort==c]
    for t in range(1,5):
        risk=z[z.end_month>=t];events=z[(z.end_month==t)&(z.event==1)]
        if len(events):sets.append((risk.training.to_numpy(),len(events),float(events.training.sum())))
enumerated=[np.array([sum(x[list(ids)]) for ids in itertools.combinations(range(len(x)),d)]) for x,d,s in sets]


def logden(beta,x,d):
    # Elementary symmetric polynomial in log space, descending degree updates.
    a=np.full(d+1,-np.inf);a[0]=0.
    for v in beta*x:
        for k in range(d,0,-1):a[k]=np.logaddexp(a[k],a[k-1]+v)
    return a[d]


def nll(beta,route='recursion'):
    total=0.
    for (x,d,s),sums in zip(sets,enumerated):
        den=logden(beta,x,d) if route=='recursion' else special.logsumexp(beta*sums)
        total+=den-beta*s
    return float(total)


opt=optimize.minimize_scalar(nll,bounds=(-4,4),method='bounded',options={'xatol':1e-12})


def result(beta,route='recursion'):
    x,d,s=sets[0];lp=beta*s-(logden(beta,x,d) if route=='recursion' else special.logsumexp(beta*enumerated[0]))
    return {'training_coefficient':float(beta),'conditional_odds_ratio':float(np.exp(beta)),'log_likelihood':-nll(beta,route),'first_event_set_probability':float(np.exp(lp)),'first_event_set_log_probability':float(lp)}


ref=result(opt.x)
n=len(df)

print(json.dumps({
    "task_id": "lp11_duration",
    "estimates": [{"term": k, "estimate": v, "se": None, "n_obs": n} for k, v in ref.items()],
}))
