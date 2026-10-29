"""Reference for two_way_cluster (OLS with standard errors clustered by firm and year) in Python.

The computation is copied from reference_two_way_cluster() in source/code/benchmark/dgp/generate_data_v3.py,
which generated the input and wrote the reference output: V = V_firm + V_year - V_firm-year (Cameron,
Gelbach, and Miller), each with the correction G/(G-1) * (N-1)/(N-K). Run from the repository root. An
optional first argument replaces the input folder tasks/two_way_cluster.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/two_way_cluster") / "pilot_v3.csv"
N_YEARS = 10

df = pd.read_csv(DATA)
y = df["y_twocluster"].values
X = sm.add_constant(df[["x"]]).values
n, k = X.shape
XtXinv = np.linalg.inv(X.T @ X)
beta = XtXinv @ (X.T @ y)
resid = y - X @ beta


def cr1_meat(groups: np.ndarray) -> np.ndarray:
    unique = np.unique(groups)
    G = len(unique)
    meat = np.zeros((k, k))
    for g in unique:
        mask = groups == g
        ug = X[mask].T @ resid[mask]
        meat += np.outer(ug, ug)
    meat *= (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    return meat


firms = df["firm"].values
years = df["year"].values
intersection = firms * (N_YEARS + 1) + years  # unique per firm-year cell
V_2way = (XtXinv @ cr1_meat(firms) @ XtXinv + XtXinv @ cr1_meat(years) @ XtXinv
          - XtXinv @ cr1_meat(intersection) @ XtXinv)
se = np.sqrt(np.diag(V_2way))
rows = [{"term": "const", "estimate": float(beta[0]), "se": float(se[0]), "n_obs": int(n)},
        {"term": "x", "estimate": float(beta[1]), "se": float(se[1]), "n_obs": int(n)}]
print(json.dumps({"estimates": rows}, indent=1))
