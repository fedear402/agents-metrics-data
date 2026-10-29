"""Reference for poisson_offset (Poisson regression with offset log(exposure)) in Python.

The model fit is copied from fit_reference() in source/code/benchmark/dgp/generate_data_v2.py, which generated the
input and wrote the reference output. Run from the repository root. An optional first argument replaces
the input folder tasks/poisson_offset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/poisson_offset") / "pilot_v2.csv"

df = pd.read_csv(DATA)
X = sm.add_constant(df[["x1", "x2"]])
offset = np.log(df["exposure"].values)
res = sm.GLM(df["y_count"], X, family=sm.families.Poisson(), offset=offset).fit()
rows = [{"term": t, "estimate": float(res.params[t]), "se": float(res.bse[t]), "n_obs": int(res.nobs)}
        for t in ("const", "x1", "x2")]
print(json.dumps({"estimates": rows}, indent=1))
