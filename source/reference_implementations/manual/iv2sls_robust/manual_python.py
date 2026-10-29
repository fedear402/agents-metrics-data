"""Reference for iv2sls_robust (2SLS with robust standard errors) in Python.

The model fit is copied from fit_reference() in source/code/benchmark/dgp/generate_data_v2.py, which generated the
input and wrote the reference output. Run from the repository root. An optional first argument replaces
the input folder tasks/iv2sls_robust.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/iv2sls_robust") / "pilot_v2.csv"

df = pd.read_csv(DATA)
res = IV2SLS(
    dependent=df["y_iv"],
    exog=sm.add_constant(pd.DataFrame(index=df.index)),  # only constant in exog
    endog=df[["x1_iv"]],
    instruments=df[["z"]],
).fit(cov_type="robust", debiased=True)
rows = [{"term": t, "estimate": float(res.params[t]), "se": float(res.std_errors[t]), "n_obs": int(res.nobs)}
        for t in ("const", "x1_iv")]
print(json.dumps({"estimates": rows}, indent=1))
