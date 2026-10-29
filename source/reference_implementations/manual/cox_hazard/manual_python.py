"""Manual reference for T13 (Cox PH) — Python."""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/cox_hazard") / "pilot_v4_cox.csv"

df = pd.read_csv(DATA)
res = PHReg(df["t_obs"].values, df[["treat", "x"]].values, status=df["event"].values).fit(method="bfgs", disp=0)
out = {
    "platform": "python",
    "coef_treat": float(res.params[0]),
    "se_treat":   float(res.bse[0]),
    "coef_x":     float(res.params[1]),
    "se_x":       float(res.bse[1]),
    "events":     int(df["event"].sum()),
    "n_obs":      int(len(df)),
}
print(json.dumps(out, indent=2))
