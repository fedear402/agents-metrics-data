"""Reference program for hard_panel_lp_iv (Python).

This is the primary route of the task build program (function panel_lp_iv):
exact calendar merges for the common origin sample, then 2SLS at each horizon
after partialling out the intercept, y_(t-1), and unit and period dummies
(ratio of instrument-residual cross-products).

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/hard_panel_lp_iv. It must hold input.csv.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/hard_panel_lp_iv")


def resid(a, c):
    return a-c@np.linalg.lstsq(c,a,rcond=None)[0]


d=pd.read_csv(INPUT / 'input.csv')
m=d.copy()
for h in [-1,1,2]:
    q=d[['unit','period','y']].copy();q.period-=h;q=q.rename(columns={'y':f'y{h}'})
    m=m.merge(q,on=['unit','period'],validate='one_to_one')
m=m.sort_values(['unit','period']); C=np.column_stack([np.ones(len(m)),m['y-1'],pd.get_dummies(m.unit,drop_first=True,dtype=float),pd.get_dummies(m.period,drop_first=True,dtype=float)])
xr=resid(m.x.to_numpy(),C); zr=resid(m.z.to_numpy(),C)
main=[float(zr@resid(m[col].to_numpy(),C)/(zr@xr)) for col in ['y','y1','y2']]
values=dict(zip(['lp_h0','lp_h1','lp_h2'],main))
n=len(m)

print(json.dumps({
    "task_id": "hard_panel_lp_iv",
    "estimates": [{"term": k, "estimate": v, "se": None, "n_obs": n} for k, v in values.items()],
}))
