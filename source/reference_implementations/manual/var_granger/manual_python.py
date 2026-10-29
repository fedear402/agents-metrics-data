"""Manual reference for T22 (Granger F-test, y2 does not Granger-cause y1, 2 lags) — Python."""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/var_granger") / "pilot_v4_var.csv"

df = pd.read_csv(DATA)
y1 = df["y1"].values
y2 = df["y2"].values
n = len(y1)
idx = np.arange(2, n)
Y = y1[idx]
X_res = sm.add_constant(np.column_stack([y1[idx - 1], y1[idx - 2]]))
X_unr = sm.add_constant(np.column_stack([y1[idx - 1], y1[idx - 2], y2[idx - 1], y2[idx - 2]]))
rss_res = float(np.sum(sm.OLS(Y, X_res).fit().resid ** 2))
rss_unr = float(np.sum(sm.OLS(Y, X_unr).fit().resid ** 2))
q = 2
k_unr = X_unr.shape[1]
n_used = len(Y)
F = ((rss_res - rss_unr) / q) / (rss_unr / (n_used - k_unr))
p_value = float(1.0 - stats.f.cdf(F, q, n_used - k_unr))
out = {
    "platform": "python",
    "F": float(F),
    "df_num": q,
    "df_den": int(n_used - k_unr),
    "p_value": p_value,
    "n_obs": int(n_used),
}
print(json.dumps(out, indent=2))
