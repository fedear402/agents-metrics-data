"""Reference for logit (logit) in Python.

The model fit is copied from fit_reference() in source/code/benchmark/dgp/generate_data.py, which generated the
input and wrote the reference output. Run from the repository root. An optional first argument replaces
the input folder tasks/logit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/logit") / "pilot.csv"

df = pd.read_csv(DATA)
X = sm.add_constant(df[["x1", "x2"]])
res = sm.Logit(df["y_bin_logit"], X).fit(disp=0)
rows = [{"term": t, "estimate": float(res.params[t]), "se": float(res.bse[t]), "n_obs": int(res.nobs)}
        for t in ("const", "x1", "x2")]
print(json.dumps({"estimates": rows}, indent=1))
