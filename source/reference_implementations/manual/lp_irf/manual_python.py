"""Manual reference for T19 (local-projection IRF at h=4 with Newey-West SE) — Python.

Estimates y_{t+h} = alpha + beta_h * shock_t + gamma * y_{t-1} + e_{t+h}
with h = 4, OLS, Newey-West (Bartlett) HAC SE with lag = h.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "pilot_v5_lp.csv"
h = 4

df = pd.read_csv(DATA).sort_values("t").reset_index(drop=True)
T = len(df)
y = df["y"].values
shock = df["shock"].values
yhp = y[h + 1:]                    # y at t = h+1 ... T-1 (zero-based -> y_{t+h}=y[i+h])
shock_t = shock[1:T - h]           # shock at t (with t-1 lag-control needing one obs back)
y_lag = y[:T - h - 1]              # y_{t-1}

X = sm.add_constant(np.column_stack([shock_t, y_lag]))
fit = sm.OLS(yhp, X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
beta_h = float(fit.params[1])
se_h = float(fit.bse[1])
out = {
    "platform": "python",
    "irf_h4": beta_h,
    "se_irf_h4": se_h,
    "horizon": h,
    "n_used": int(len(yhp)),
}
print(json.dumps(out, indent=2))
