"""Manual reference for T21 (VAR(2) orthogonal IRF, y1 response to y2 shock, h=4) — Python."""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
from statsmodels.tsa.api import VAR

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/var_irf") / "pilot_v4_var.csv"

df = pd.read_csv(DATA)
Y = df[["y1", "y2"]].values
res = VAR(Y).fit(2)
irf = res.irf(periods=10).orth_irfs   # (h+1, response, shock)
val = float(irf[4, 0, 1])              # response y1 (idx 0), shock y2 (idx 1)
out = {
    "platform": "python",
    "irf_y1_to_y2_h4": val,
    "lag_order": 2,
    "identification": "Cholesky, y1 ordered first",
    "n_obs": int(len(df)),
}
print(json.dumps(out, indent=2))
