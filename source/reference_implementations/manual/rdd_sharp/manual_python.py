"""Manual reference for T18 (sharp regression discontinuity, local linear) — Python.

Uses a triangular kernel with a fixed bandwidth h = 0.5. Estimates the LATE at
the cutoff z=0 as the difference between two local-linear regressions on either
side.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v5_rdd.csv"
h = 0.5

df = pd.read_csv(DATA)
z = df["z"].values
y = df["y"].values
mask = np.abs(z) < h
zb = z[mask]
yb = y[mask]
w = (1.0 - np.abs(zb) / h)   # triangular kernel

def wls_intercept(z_side, y_side, w_side):
    X = np.column_stack([np.ones_like(z_side), z_side])
    W = np.diag(w_side)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y_side
    coef = np.linalg.solve(XtWX, XtWy)
    return float(coef[0])  # intercept = level at z=0

right = zb >= 0
left = ~right
mu_right = wls_intercept(zb[right], yb[right], w[right])
mu_left = wls_intercept(zb[left], yb[left], w[left])
late = mu_right - mu_left

# n_obs is the effective estimation sample (observations inside the bandwidth),
# which is what the benchmark result schema records and what every model reports.
# The full input file has len(df) rows; that count is reported separately.
out = {
    "platform": "python",
    "late": late,
    "bandwidth": h,
    "n_inside_band": int(mask.sum()),
    "n_file_rows": int(len(df)),
    "n_obs": int(mask.sum()),
}
print(json.dumps(out, indent=2))
