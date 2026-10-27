"""Manual reference for T14 (two-way FE OLS with cluster-robust SE) — Python.

Estimator: regress y_twocluster on x with firm and year fixed effects absorbed,
report the x coefficient with cluster-robust SE on firm. Uses the within
transformation (double-demeaning, iterated) and the standard CR1 cluster-robust
sandwich.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v3.csv"

df = pd.read_csv(DATA)
y = df["y_twocluster"].values
x = df["x"].values
firm = df["firm"].values
year = df["year"].values

# Iterative within transformation (Gauss-Seidel for two FE) — converges to the
# two-way demeaning identity.
def demean_two_way(z, g1, g2, n_iter=50):
    z = z.copy().astype(float)
    for _ in range(n_iter):
        for g in (g1, g2):
            means = pd.Series(z).groupby(g).transform("mean").values
            z = z - means
    return z

y_dm = demean_two_way(y, firm, year)
x_dm = demean_two_way(x, firm, year)

beta = float(np.sum(x_dm * y_dm) / np.sum(x_dm * x_dm))
resid = y_dm - beta * x_dm
n = len(y)

# Cluster-robust on firm (CR1 with G/(G-1)*(N-1)/(N-K) correction).
firms = np.unique(firm)
G = len(firms)
score = np.zeros(1)
meat = 0.0
for g in firms:
    mask = firm == g
    s = float(np.sum(x_dm[mask] * resid[mask]))
    meat += s * s
# bread = sum x_dm^2 (denominator from OLS)
bread = float(np.sum(x_dm * x_dm))
# Account for two-way FE: 200 firm dummies + 10 year dummies + 1 intercept absorbed = 211, plus x = 212.
# Use a stylized K = number of absorbed FE + slope to keep cross-platform comparable.
K = 200 + 10 + 1
V = (meat / (bread * bread)) * (G / (G - 1.0)) * ((n - 1.0) / (n - K))
se = float(np.sqrt(V))
out = {"platform": "python", "x": beta, "se_x_cluster_firm": se, "n_obs": int(n), "n_firms": int(G), "K_absorbed": int(K)}
print(json.dumps(out, indent=2))
