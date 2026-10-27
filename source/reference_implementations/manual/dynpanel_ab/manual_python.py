"""Manual reference for T16 (Arellano-Bond first-difference GMM) — Python.

One-step first-difference GMM with the standard lagged-level instruments. For
the model
    y_{i,t} = rho * y_{i,t-1} + beta * x_{i,t} + alpha_i + eps_{i,t},
the moment conditions are E[y_{i,s} * eps_{i,t} - eps_{i,t-1}] = 0 for s <= t-2.
The instrument matrix is constructed per-i, lower-triangular with available
y_{i,s} for s <= t-2. x is treated as strictly exogenous (instrument by itself
in first differences).
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v5_dynpanel.csv"

df = pd.read_csv(DATA).sort_values(["firm", "year"]).reset_index(drop=True)
firms = df["firm"].unique()
T = int(df["year"].max() + 1)

# Build first-difference stack for t = 2..T-1 (so we have y_{t-1} and y_{t-2} available).
dy_list, dy_lag_list, dx_list, Z_list = [], [], [], []
n_inst = sum(range(T - 2 + 1))  # max columns; we'll pad with zeros for missing
for i in firms:
    yi = df.loc[df["firm"] == i, "y"].values
    xi = df.loc[df["firm"] == i, "x"].values
    for t in range(2, T):
        dy = yi[t] - yi[t - 1]
        dy_lag = yi[t - 1] - yi[t - 2]
        dx = xi[t] - xi[t - 1]
        # Instruments: y_{i,0}, y_{i,1}, ..., y_{i,t-2}, plus dx itself.
        n_y_inst = t - 1   # y_{i,0..t-2}
        z_y = np.zeros(T - 2)
        z_y[:n_y_inst] = yi[:n_y_inst]
        z = np.concatenate([z_y, [dx]])
        dy_list.append(dy)
        dy_lag_list.append(dy_lag)
        dx_list.append(dx)
        Z_list.append(z)

dY = np.array(dy_list)
dYl = np.array(dy_lag_list)
dX = np.array(dx_list)
Z = np.array(Z_list)
X = np.column_stack([dYl, dX])

# One-step GMM with identity-like weighting: W = (Z'Z)^{-1}.
ZtZ_inv = np.linalg.pinv(Z.T @ Z)
XZ = X.T @ Z
beta = np.linalg.solve(XZ @ ZtZ_inv @ XZ.T, XZ @ ZtZ_inv @ Z.T @ dY)
resid = dY - X @ beta
# Standard errors (one-step, homoskedastic): V = (XZ W ZX)^{-1} XZ W (Z' E E' Z) W ZX (XZ W ZX)^{-1}
# where E is the residual vector; we use the standard one-step formula.
E = resid
ZE = Z.T @ E
S = np.outer(ZE, ZE)
W = ZtZ_inv
A = XZ @ W @ XZ.T
V = np.linalg.solve(A, np.eye(2)) @ XZ @ W @ S @ W @ XZ.T @ np.linalg.solve(A, np.eye(2))
se = np.sqrt(np.diag(V))
out = {
    "platform": "python",
    "rho": float(beta[0]),
    "se_rho": float(se[0]),
    "beta_x": float(beta[1]),
    "se_beta_x": float(se[1]),
    "n_obs": int(len(dY)),
    "n_firms": int(len(firms)),
}
print(json.dumps(out, indent=2))
