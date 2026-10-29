"""Manual reference for T08 (one-way ANOVA) — Python."""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/anova_oneway") / "pilot_v5_anova.csv"

df = pd.read_csv(DATA)
groups = [df.loc[df["group"] == g, "y"].values for g in sorted(df["group"].unique())]
F, p = stats.f_oneway(*groups)
out = {
    "platform": "python",
    "F": float(F),
    "df_num": int(len(groups) - 1),
    "df_den": int(len(df) - len(groups)),
    "p_value": float(p),
    "n_obs": int(len(df)),
}
print(json.dumps(out, indent=2))
