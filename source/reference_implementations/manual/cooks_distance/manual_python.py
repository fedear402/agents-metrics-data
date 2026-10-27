"""Manual reference for T06 (Cook's distance, top-5 influential observations) — Python."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v5_collinear.csv"

df = pd.read_csv(DATA)
y = df["y"].values
X = np.c_[np.ones(len(df)), df[["x1", "x2", "x3", "x4", "x5"]].values]
n, k = X.shape
beta = np.linalg.solve(X.T @ X, X.T @ y)
yhat = X @ beta
resid = y - yhat
H = X @ np.linalg.solve(X.T @ X, X.T)
h = np.diag(H)
ss_res = float(np.sum(resid ** 2))
sigma2 = ss_res / (n - k)
cooks_d = (resid ** 2 / (k * sigma2)) * (h / (1 - h) ** 2)
top5 = np.argsort(-cooks_d)[:5]
out = {
    "platform": "python",
    "top5_obs_ids": [int(i) for i in top5.tolist()],
    "top5_cooks_d": [float(cooks_d[i]) for i in top5],
    "n_obs": int(n),
}
print(json.dumps(out, indent=2))
