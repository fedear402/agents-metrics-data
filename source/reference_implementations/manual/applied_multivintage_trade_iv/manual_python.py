"""Reference program for applied_multivintage_trade_iv (Python).

This is the reference route of the task build program (function a1):
long-table weighted exposure sums and projected weighted 2SLS that includes
all exogenous columns in the instrument matrix.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/applied_multivintage_trade_iv. It must hold
areas.csv, employment.csv, trade.csv, crosswalk.csv, and outcomes.csv.
The program prints one JSON object and writes no files.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/applied_multivintage_trade_iv")


def iv(X, Z, y, w):
    """Weighted projection 2SLS, including all exogenous columns in Z."""
    s = np.sqrt(w)
    a, b, c = X*s[:,None], Z*s[:,None], y*s
    fitted = b @ np.linalg.lstsq(b, a, rcond=None)[0]
    return np.linalg.solve(fitted.T @ a, fitted.T @ c)


a = pd.read_csv(INPUT / 'areas.csv')
e = pd.read_csv(INPUT / 'employment.csv')
t = pd.read_csv(INPUT / 'trade.csv')
cross = pd.read_csv(INPUT / 'crosswalk.csv')
o = pd.read_csv(INPUT / 'outcomes.csv')


def exposure(kind, vintage, matrix=False, denominator='total'):
    ans=[]
    for q in (0,1):
        ee=e[(e.period==q)&(e.vintage==vintage)]
        aa=a[(a.period==q)&(a.vintage==vintage)].sort_values('area')
        tt=t[t.period==q].merge(cross,on='code').sort_values('industry')
        national=ee.groupby('industry').employment.sum().reindex(range(4)).to_numpy()
        shocks=tt[kind+'_change'].to_numpy()/(tt.price_index.to_numpy()/100)/1000*10/tt.years.to_numpy()/national
        mat=ee.pivot(index='area',columns='industry',values='employment').reindex(index=range(36),columns=range(4)).fillna(0).to_numpy()
        denom=aa.total_employment.to_numpy() if denominator=='total' else mat.sum(axis=1)
        if matrix: ans.extend((mat@shocks/denom).tolist())
        else:
            lookup=dict(zip(tt.industry,shocks))
            ans.extend([sum(row.employment*lookup[row.industry] for row in ee[ee.area==i].itertuples())/denom[i] for i in range(36)])
    return np.array(ans)


x=exposure('us','base'); z=exposure('other','early')
w=a[a.vintage=='base'].sort_values(['period','area']).population.to_numpy(float)
C=np.column_stack([np.ones(72),o.period,o.control]); X=np.column_stack([C,x]); Z=np.column_stack([C,z])
b=iv(X,Z,o.change_pp.to_numpy(),w)
main={'iv_slope':float(b[-1]),'exposure_area0_period0':float(x[0]),'exposure_area0_period1':float(x[36]),'instrument_area0_period0':float(z[0])}
n = 72

print(json.dumps({
    "task_id": "applied_multivintage_trade_iv",
    "estimates": [{"term": k, "estimate": v, "se": None, "n_obs": n} for k, v in main.items()],
}))
