"""Manual reference for T02 (ARIMA(1,1,1) one-step forecast) — Python."""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/arima_forecast") / "pilot_v4_arima.csv"

df = pd.read_csv(DATA)
y = df["y"].values
res = ARIMA(y, order=(1, 1, 1)).fit()
fc = res.get_forecast(steps=1)
mean = float(fc.predicted_mean[0])
se = float(fc.se_mean[0])
ci = fc.conf_int(alpha=0.05)[0]
out = {
    "platform": "python",
    "forecast": mean,
    "se": se,
    "lower_95": float(ci[0]),
    "upper_95": float(ci[1]),
    "n_obs": int(len(y)),
}
print(json.dumps(out, indent=2))
