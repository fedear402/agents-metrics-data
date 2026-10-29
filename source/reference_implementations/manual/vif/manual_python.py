"""Manual reference for T05 (VIF for a 5-variable linear model) — Python.

VIF_j = 1 / (1 - R^2_j), where R^2_j is the R-squared from regressing x_j on
the other four regressors with an intercept. Returns VIF for x1..x5.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/vif") / "pilot_v5_collinear.csv"

df = pd.read_csv(DATA)
cols = ["x1", "x2", "x3", "x4", "x5"]
vif = {}
for j, c in enumerate(cols):
    others = [o for o in cols if o != c]
    X = np.c_[np.ones(len(df)), df[others].values]
    y = df[c].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    vif[c] = 1.0 / (1.0 - r2)

out = {"platform": "python", **{f"vif_{c}": v for c, v in vif.items()}, "n_obs": int(len(df))}
print(json.dumps(out, indent=2))
