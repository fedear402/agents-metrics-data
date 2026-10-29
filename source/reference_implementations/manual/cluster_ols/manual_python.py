"""Reference for cluster_ols (OLS with CR1 standard errors clustered by cluster) in Python.

The model fit is copied from fit_reference() in source/code/benchmark/dgp/generate_data_v2.py, which generated the
input and wrote the reference output. Run from the repository root. An optional first argument replaces
the input folder tasks/cluster_ols.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/cluster_ols") / "pilot_v2.csv"

df = pd.read_csv(DATA)
X = sm.add_constant(df[["x1", "x2"]])
res = sm.OLS(df["y_panel"], X).fit(cov_type="cluster", cov_kwds={"groups": df["cluster"].values})
rows = [{"term": t, "estimate": float(res.params[t]), "se": float(res.bse[t]), "n_obs": int(res.nobs)}
        for t in ("const", "x1", "x2")]
print(json.dumps({"estimates": rows}, indent=1))
