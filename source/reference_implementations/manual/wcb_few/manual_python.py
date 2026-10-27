"""Manual reference for T20 (wild cluster bootstrap, 5 clusters, B=999) — Python.

OLS of y on x with cluster-robust CR1 SE on cluster. Bootstrap the t-statistic
under H0: beta = 0 by Rademacher weights at the cluster level. Report the
two-sided p-value as the share of |t*| >= |t_obs|.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v5_wcb.csv"

df = pd.read_csv(DATA)
y = df["y"].values
x = df["x"].values
cluster = df["cluster"].values
n = len(y)

X = np.column_stack([np.ones(n), x])
XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ (X.T @ y)
resid = y - X @ beta
clusters = np.unique(cluster)
G = len(clusters)
K = X.shape[1]

def cr1_se(resid_e, X_e):
    meat = np.zeros((K, K))
    for g in clusters:
        mask = cluster == g
        Xg = X_e[mask]; eg = resid_e[mask]
        ug = Xg.T @ eg
        meat += np.outer(ug, ug)
    correction = (G / (G - 1.0)) * ((n - 1.0) / (n - K))
    V = correction * XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.diag(V))

se = cr1_se(resid, X)
t_obs = float(beta[1] / se[1])

# Impose H0: beta_x = 0 by regressing y on intercept only and bootstrapping
# residuals from the restricted model.
beta_r = np.array([y.mean(), 0.0])
resid_r = y - X @ beta_r
rng = np.random.default_rng(20260520)
B = 999
extreme = 0
for _ in range(B):
    weights = rng.choice([-1.0, 1.0], size=G)
    star = resid_r * weights[np.searchsorted(clusters, cluster)]
    y_star = X @ beta_r + star
    beta_star = XtX_inv @ (X.T @ y_star)
    resid_star = y_star - X @ beta_star
    se_star = cr1_se(resid_star, X)
    t_star = (beta_star[1] - 0.0) / se_star[1]
    if abs(t_star) >= abs(t_obs):
        extreme += 1
p_value = float(extreme / B)
out = {
    "platform": "python",
    "x": float(beta[1]),
    "se_x_cr1": float(se[1]),
    "t_obs": t_obs,
    "p_wcb": p_value,
    "B": B,
    "n_clusters": int(G),
    "n_obs": int(n),
}
print(json.dumps(out, indent=2))
