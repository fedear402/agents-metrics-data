"""Reference program for broad_oaxaca (Python).

This is the reference computation of the task review (the broad_oaxaca branch
of its shared reference module). It fits separate least-squares regressions of
y on an intercept, x, and z in each group with NumPy and forms the
decomposition with group 0 coefficients as the reference. The stored reference
is this route. The independent route of the same module (statsmodels OLS) is
printed as an extra key.

Run from the data repository root:
    python manual_python.py [INPUT_DIR]
INPUT_DIR defaults to tasks/broad_oaxaca.
The program prints one JSON object and writes no files.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/broad_oaxaca")
df = pd.read_csv(INPUT / "input.csv")

a=df[df.group==0]; b=df[df.group==1]
X0=sm.add_constant(a[['x','z']]).to_numpy(); X1=sm.add_constant(b[['x','z']]).to_numpy()
def calculate(b0,b1):
    d=X1.mean(0)-X0.mean(0)
    return dict(gap=b.y.mean()-a.y.mean(),explained=d@b0,unexplained=X1.mean(0)@(b1-b0),explained_x=d[1]*b0[1])
current=calculate(np.linalg.lstsq(X0,a.y,rcond=None)[0],np.linalg.lstsq(X1,b.y,rcond=None)[0])
independent=calculate(sm.OLS(a.y,X0).fit().params.to_numpy(),sm.OLS(b.y,X1).fit().params.to_numpy())
diagnostics={'identity_error':float(current['gap']-current['explained']-current['unexplained']),'reference_group':0,'group_sizes':[len(a),len(b)]}
assert max(abs(current[k]-independent[k]) for k in current)<2e-5

result = {
    "estimates": [
        {"term": term, "estimate": float(current[term]), "se": None, "n_obs": int(len(df))}
        for term in ["gap", "explained", "unexplained", "explained_x"]
    ],
    "independent_route": {k: float(v) for k, v in independent.items()},
    "diagnostics": diagnostics,
}
print(json.dumps(result, indent=2))
