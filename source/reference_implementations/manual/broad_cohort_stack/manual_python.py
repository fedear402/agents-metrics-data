"""Reference program for broad_cohort_stack (Python).

This is the reference computation of the task review (the broad_cohort_stack
branch of its shared reference module). For each adoption cohort it keeps the
five-period window, the clean controls, and the balanced units. It then
estimates the did coefficient by balanced-panel double demeaning. The stored
reference is this route. The independent route of the same module (least
squares with full unit and time dummies) is printed as an extra key.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/broad_cohort_stack.
The program prints one JSON object and writes no files.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/broad_cohort_stack")
df = pd.read_csv(INPUT / "input.csv")

df=df.rename(columns={'time':'t'})
current={};independent={};weights=[];diagnostics={};rows=0
for c in [3,5,7]:
    g=df[df.t.between(c-2,c+2)&((df.g==c)|(df.g==0)|(df.g>c+2))].copy()
    ids=g.groupby('id').t.nunique();g=g[g.id.isin(ids[ids==5].index)].copy()
    g['did']=((g.g==c)&(g.t>=c)).astype(float)
    y=g.y-g.groupby('id').y.transform('mean')-g.groupby('t').y.transform('mean')+g.y.mean()
    x=g.did-g.groupby('id').did.transform('mean')-g.groupby('t').did.transform('mean')+g.did.mean()
    current[f'att_g{c}']=float(x@y/(x@x))
    X=pd.concat([g[['did']].reset_index(drop=True),pd.get_dummies(g.id,drop_first=True,dtype=float).reset_index(drop=True),pd.get_dummies(g.t,drop_first=True,dtype=float).reset_index(drop=True)],axis=1)
    independent[f'att_g{c}']=float(np.linalg.lstsq(sm.add_constant(X).to_numpy(),g.y.to_numpy(),rcond=None)[0][1])
    ntr=g.loc[g.g==c,'id'].nunique();weights.append(ntr);rows+=len(g);diagnostics[str(c)]={'treated_units':ntr,'rows':len(g)}
current['att_weighted']=np.average([current[f'att_g{c}'] for c in [3,5,7]],weights=weights)
independent['att_weighted']=np.average([independent[f'att_g{c}'] for c in [3,5,7]],weights=weights)
diagnostics['summed_comparison_rows']=rows
assert max(abs(current[k]-independent[k]) for k in current)<2e-5

n_obs = {f"att_g{c}": diagnostics[str(c)]["rows"] for c in [3, 5, 7]}
n_obs["att_weighted"] = diagnostics["summed_comparison_rows"]
result = {
    "estimates": [
        {"term": term, "estimate": float(current[term]), "se": None, "n_obs": int(n_obs[term])}
        for term in ["att_g3", "att_g5", "att_g7", "att_weighted"]
    ],
    "independent_route": {k: float(v) for k, v in independent.items()},
    "diagnostics": diagnostics,
}
print(json.dumps(result, indent=2))
